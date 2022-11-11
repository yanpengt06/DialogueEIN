import numpy as np
import torch


def person_embed(speaker_ids, person_vec):
    '''

    :param speaker_ids: torch.Tensor ( T, B)
    :param person_vec: numpy array (num_speakers, 100)
    :return:
        speaker_vec: torch.Tensor (T, B, D)
    '''
    speaker_vec = []
    for t in speaker_ids:
        speaker_vec.append([person_vec[int(i)].tolist() if i != -1 else [0] * 100 for i in t])
    speaker_vec = torch.FloatTensor(speaker_vec)
    return speaker_vec


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    return ~mask
