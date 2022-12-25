import json

import torch


def test(model, rbt, tokenizer, utts, speakers, device):
    """
    @params:
    model: dialogueEIN trained on JDDC
    rbt: finetune on JDDC
    utts: List[String] [u1, u2, ..., uT]
    speakers: List[int] [0,1,0,0,1, ...] 需要映射到int
    """
    features = get_cls(rbt, tokenizer, utts, device)   # floatTensor: T x H(768)
    features = features.unsqueeze(0) # 1 x T x H
    speakers = torch.LongTensor(speakers).unsqueeze(0).to(device) # 1 x T
    lengths = torch.LongTensor([speakers.shape[1]]).to(device)
    logits = model(features, lengths, speakers) # 1 x T x C (C=3) ["中性", "积极", "消极"]
    preds = torch.argmax(logits, dim=-1).squeeze(0).tolist() # list: [T]
    return preds


def get_cls(rbt, tokenizer, text, device, avg=False):
    encoded_text = tokenizer(text, return_tensors='pt', padding=True)
    encoded_text.to(device)
    with torch.no_grad():
        output = rbt(**encoded_text, output_hidden_states=True)
    B, H = output.last_hidden_state.shape[0], output.last_hidden_state.shape[2]
    if avg is False:
        return output.last_hidden_state[:, 0]  # cls feature
    else:
        result = torch.zeros((B, H)).to(device)
        for i in range(9, 13):
            result += output.hidden_states[i][:, 0]  # cls feature in i-th layer (768) # B x T x H -> B x H
        result /= 4
        # print(result.shape)
        return result


if __name__ == '__main__':
    from transformers import AutoTokenizer, BertModel

    with open("./data/jddc/train_data_roberta_v2.json.feature", "r", encoding='utf-8') as f:
        dialogs = json.load(f)
        dialog1 = dialogs[0]
        utts = []
        speakers = []
        features = []
        labels = []
        for utt in dialog1:
            utts.append(utt["text"])
            features.append(utt["cls"])
            speakers.append(utt["speaker"])
            labels.append(utt["label"])
        assert len(features) == len(speakers) == len(utts)
        dlg = {}
        speakers = [0 if speaker == 'A' else 1 for speaker in speakers]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"current device on {device}")
    rbt = BertModel.from_pretrained("./checkpoint-147")
    tokenizer = AutoTokenizer.from_pretrained("./checkpoint-147")
    model = torch.load("./EIN-10-95.6.pkl")
    rbt = rbt.to(device)
    model = model.to(device)
    print("start to inference...")
    preds = test(model, rbt, tokenizer, utts, speakers, device)
    print("Ture labels:", sep='')
    print(labels)
    print("predicted results:", sep='')
    print(preds)
