import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
from transformers import BertModel, BertConfig, AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelWithLMHead
from model_utils import *
from utils import sequence_mask

logger = logging.getLogger(__name__)


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=384,
                 num_hidden_layers=12,
                 num_attention_heads=8,
                 intermediate_size=1024,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 dataset_name='IEMOCAP'):
        """Constructs BertConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):  # or (sys.version_info[0] == 2
            # and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.dataset_name = dataset_name
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

"""BERT model ("Bidirectional Embedding Representations from a Transformer").
Params:
    config: a BertConfig class instance with the configuration to build a new model
Inputs:
    `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
        with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
        `extract_features.py`, `run_classifier.py` and `run_squad.py`)
    `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
        types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
        a `sentence B` token (see BERT paper for more details).
    `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
        selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
        input sequence length in the current batch. It's the mask that we typically use for attention when
        a batch has varying length sentences.
    `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
Outputs: Tuple of (encoded_layers, pooled_output)
    `encoded_layers`: controled by `output_all_encoded_layers` argument:
        - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
            of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
            encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
        - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
            to the last attention block of shape [batch_size, sequence_length, hidden_size],
    `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
        classifier pretrained on top of the hidden state associated to the first character of the
        input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
Example usage:
```python
# Already been converted into WordPiece token ids
input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
model = modeling.BertModel(config=config)
all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
```
"""


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q_input, k_input, v_input, attention_mask):
        mixed_query_layer = self.query(q_input)
        mixed_key_layer = self.key(k_input)
        mixed_value_layer = self.value(v_input)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EmotionSelfOutput(nn.Module):
    def __init__(self, config):
        super(EmotionSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, q_input, k_input, v_input, attention_mask):
        self_output = self.self(q_input, k_input, v_input, attention_mask)
        attention_output = self.output(self_output, q_input)
        return attention_output


class EmotionAttention(nn.Module):
    def __init__(self, config):
        super(EmotionAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = EmotionSelfOutput(config)

    def forward(self, q_input, k_input, v_input, attention_mask):
        self_output = self.self(q_input, k_input, v_input, attention_mask)
        attention_output = self.output(self_output)
        return attention_output


class DialogueEIN(nn.Module):
    """reproduction of DialogueEIN"""

    def __init__(self, config, emotion_num, window_size, device):
        super(DialogueEIN, self).__init__()
        self.device = device
        if config.dataset_name in ['IEMOCAP']:  # roberta-base
            config.hidden_size = 384
            self.roberta_dim = 768
            self.roberta = AutoModelForMaskedLM.from_pretrained("roberta-base")
        elif config.dataset_name in ['MELD', 'jddc']:
            config.hidden_size = 512
            self.roberta = AutoModelForMaskedLM.from_pretrained("roberta-large")
            # self.roberta = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext-large") # train on jddc, chinese-roberta-large
            self.roberta_dim = 1024
            config.num_hidden_layers = 24
        self.semantic_encoder = SemanticEncoder(config.hidden_size)
        self.emo_num = emotion_num
        self.window_size = window_size
        self.emotion_ebd = nn.Embedding(emotion_num, config.hidden_size)
        self.tendency_mha = BertAttention(config)
        self.global_mha = EmotionAttention(config)
        self.local_mha = EmotionAttention(config)
        self.inter_mha = EmotionAttention(config)
        self.intra_mha = EmotionAttention(config)
        self.transform1 = nn.Linear(4 * config.hidden_size, config.hidden_size)  # Emotion interaction layer
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.transform2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)  # classify layer

        self.linear_trans = nn.Linear(self.roberta_dim, config.hidden_size)
        self.mlp = MLP(2, self.roberta_dim, 512, emotion_num)

        self.config = config

    def forward(self, utts, att_mask, lengths, speakers):
        """
        @params:
        utts: B x S x W, S is the max dialogue length, W is the max sentence length, which is 512, as roberta-base limits
        speakers: B x T  speaker encoding, T has the same meaning as S
        lengths: B dialogue length
        att_mask: B x S x W, transformers attention_mask tensor
        """
        B = utts.shape[0]
        S = utts.shape[1]
        utts = utts.view(-1, utts.shape[-1]) # (B x S, W)
        att_mask = att_mask.view(-1, att_mask.shape[-1]) # (B x S, W)
        output = self.roberta(input_ids=utts, attention_mask=att_mask, output_hidden_states=True) # output object
        features = output["hidden_states"][self.config.num_hidden_layers][:,0] # (B x S, H) get the cls feature
        features = features.view(B,S,features.shape[-1]) # B x T x H
        features = self.linear_trans(features)  # B x T x U, U is feature size of an utterance
        h_s = self.semantic_encoder(features, lengths)  # B x T x U
        att_mask = torch.ones(1, self.emo_num)  # 1 x emo_num (broadcast to B x 1 x 1 x emo_num )
        extended_att_mask = att_mask.unsqueeze(1).unsqueeze(2).to(self.device)  # 1 x 1 x 1 x T
        # .to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_att_mask = (1.0 - extended_att_mask) * -10000.0
        h_e = self.tendency_mha(h_s, self.emotion_ebd.weight.unsqueeze(0), self.emotion_ebd.weight.unsqueeze(0),
                                extended_att_mask)  # B x T x U
        h_a_global = self.global_mha(h_e, h_e, h_e, get_ext_att_mask(lengths, device=self.device))  # B x T x H
        h_a_local = self.local_mha(h_e, h_e, h_e, get_ext_att_mask(lengths, window_size=self.window_size, type="local",
                                                                   device=self.device))
        h_a_inter = self.inter_mha(h_e, h_e, h_e,
                                   get_ext_att_mask(lengths, type="inter", speakers=speakers, device=self.device))
        h_a_intra = self.intra_mha(h_e, h_e, h_e,
                                   get_ext_att_mask(lengths, type="intra", speakers=speakers, device=self.device))
        h_a = torch.cat([h_a_global, h_a_local, h_a_inter, h_a_intra], dim=-1)  # B x T x 4U
        h_a = self.transform1(h_a)  # B x T x U
        h_a = self.LayerNorm(h_a + h_s)  # B x T x U
        h_a = self.transform2(h_a)  # B x T x U
        logits = torch.matmul(h_a, self.emotion_ebd.weight.transpose(0, 1).unsqueeze(0))  # B x T x C
        return logits
        # features = features.view(-1, features.shape[-1]) # (B x T, H)
        # return self.mlp(features) # B x T x C


def get_ext_att_mask(lengths, window_size=5, type: str = "global", speakers=None, device="cpu"):
    """
    @params:
    lengths: B
    speakers: B x T, T is the max_len of a batch data
    """
    maxlen = torch.max(lengths).item()
    # global att, normal_mask
    if type == "global":

        mask = torch.arange((maxlen), dtype=torch.float32)[None, :].to(device) < lengths[:, None]
        mask = mask + 0
        mask = mask.unsqueeze(1).unsqueeze(2).to(device)
    elif type == "local":
        B = lengths.shape[0]
        matrix_list = []
        for i in range(B):
            att_matrix = torch.zeros(maxlen, maxlen)
            for j in range(lengths[i]):  # j-th utt
                if window_size % 2 != 0:
                    start = max(0, j - int(window_size / 2))
                    end = min(lengths[i] - 1, j + int(window_size / 2))
                else:
                    start = max(0, (j - window_size / 2 - 1))
                    end = min(lengths[i] - 1, j + (j - window_size / 2 + 1))
                att_matrix[j, start:end + 1] = 1
            matrix_list.append(att_matrix)
        mask = torch.stack(matrix_list)
        mask = mask.unsqueeze(1).to(device)
    elif type == "intra":
        B = lengths.shape[0]
        matrix_list = []
        for i in range(B):
            att_matrix = torch.zeros(maxlen, maxlen)  # T x T
            i_speakers = speakers[i]
            for j in range(lengths[i]):
                utt_speaker = speakers[i][j]
                mask = i_speakers == utt_speaker
                mask = mask + 0
                att_matrix[j, :] = mask
            matrix_list.append(att_matrix)
        mask = torch.stack(matrix_list).unsqueeze(1).to(device)
    elif type == "inter":
        B = lengths.shape[0]
        matrix_list = []
        for i in range(B):
            att_matrix = torch.zeros(maxlen, maxlen)  # T x T
            i_speakers = speakers[i]
            for j in range(lengths[i]):
                utt_speaker = speakers[i][j]
                mask = (i_speakers != utt_speaker)
                mask = mask + 0
                att_matrix[j, :] = mask
            att_matrix[:, lengths[i]:] = 0
            matrix_list.append(att_matrix)
        mask = torch.stack(matrix_list).unsqueeze(1).to(device)
    return (1.0 - mask) * -10000.


class SemanticEncoder(nn.Module):
    """
    default: nhead=8, dim_ffn = 2048, dropout = 0.1
    """

    def __init__(self, utt_dim):
        super(SemanticEncoder, self).__init__()
        self.utt_dim = utt_dim
        self.positional_encoding = PositionalEncoding(utt_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=utt_dim, nhead=8, batch_first=True)
        self.semantic_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, features, lengths):
        """
        @params:
        features: B x M x H, M: max sequence length
        lengths: B
        @return:
        h_s: B x M x H
        """
        utt_mask = sequence_mask(features, lengths)
        features = self.positional_encoding(features)
        h_s = self.semantic_encoder(features, src_key_padding_mask=utt_mask)
        return h_s


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 对偶数位置编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 对奇数位置编码
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 不对位置编码层求梯度

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]  # 输入的词嵌入与位置编码相加
        return x


class MLP(nn.Module):

    def __init__(self, layer_num, input_dim, hidden_dim, class_num):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, class_num)
        self.hidden_layers = nn.ModuleList([copy.deepcopy(self.hidden_layer) for _ in range(layer_num - 1)])

    def forward(self, inputs):
        """
        @params:
        inputs B x input_dim
        @returns:
        logits B x C
        """
        inputs = self.input_layer(inputs)  # B x H
        for hidden_layer in self.hidden_layers:
            inputs = hidden_layer(inputs)  # B x H
        outputs = self.output_layer(inputs)  # B x C, logits
        return outputs


if __name__ == '__main__':
    features = torch.randn(3, 4, 768)  # B x T x H
    length = torch.LongTensor([4, 2, 3])
    speakers = torch.LongTensor([[0, 1, 0, 1], [0, 1, -1, -1], [0, 1, 0, -1]])
    config = BertConfig(1)
    EIN = DialogueEIN(config, 6)
    logits = EIN(features, length, speakers)
    print(logits)
