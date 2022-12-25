import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForMaskedLM


class IEMOCAPDataset(Dataset):

    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None,
                 tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)

        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        # adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        # s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)  # B x N
        utterances = [d[4] for d in data]

        return feaures, labels, speakers, lengths, utterances


class IEMOCAPDataset2(Dataset):

    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None,
                 tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        print(f"dataset size is:{len(self.data)}")
        # self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        if dataset_name in ['MELD', 'EmoryNLP']:
            version = 9
        else:
            version = 2

        with open(f'./data/%s/%s_data_roberta_v{version}.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)

        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            label
            speaker
            length
            utterances
        '''
        return torch.LongTensor(self.data[index]['labels']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        '''
        :param data:
            a list of dialogues, d: (labels, speakers, lengths, utterances)
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''

        labels = pad_sequence([d[0] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        lengths = torch.LongTensor([d[2] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[1]) for d in data], batch_first=True, padding_value=-1)  # B x N
        utterances = [d[3] for d in data]
        utts = []
        att_mask = []
        for d in data:
            encoded_utts = self.tokenizer(d[3], padding='max_length', truncation=True, return_tensors='pt',
                                          max_length=32)
            utts.append(encoded_utts["input_ids"])  # S x M
            att_mask.append(encoded_utts["attention_mask"])  # S x M
        utts = pad_sequence(utts, batch_first=True, padding_value=1)  # <pad> -> 1 # B x S x M
        att_mask = pad_sequence(att_mask, batch_first=True, padding_value=0)  # B x S x M
        return labels, speakers, lengths, utterances, utts, att_mask
