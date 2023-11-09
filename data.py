import librosa
import numpy as np
import random 
from utils import normalize_audio
import torch
import os
import json 
import torch.utils.data as data
from torch.utils.data import Dataset,DataLoader

def load_mixtures_and_sources(batch,sample_rate=8000,L=16):
    """
    Returns:
        mixtures: a list containing B items, each item is K x L np.ndarray
        sources: a list containing B items, each item is K x L x C np.ndarray
        K varies from item to item.
    """
    mixtures, sources = [], []

    for s1_info, s2_info in batch:

        s1_path = s1_info
        s2_path = s2_info
        # read wav file

        s1, sample_rate = librosa.load(s1_path, sr=None)
        s2, sample_rate = librosa.load(s2_path, sr=None)

        
        def cut_n_pad(arr,mix_len):
            if(len(arr)<mix_len):
                return np.concatenate([arr, np.zeros([mix_len - len(arr)])])
            return arr[:mix_len]
        
        pad_len =  min(len(s1),len(s2),3*sample_rate)
        pad_s1 = cut_n_pad(s1,pad_len)
        pad_s2 = cut_n_pad(s2,pad_len)

        pad_s1 = normalize_audio(pad_s1,sample_rate)
        pad_s2 = normalize_audio(pad_s2,sample_rate)
        sir = random.uniform(0,2.5)
        weight = 10 * ((sir) / 20)
        pad_s1 = weight * pad_s1
        pad_mix = pad_s1 + pad_s2

        
        # K = int(np.ceil(mix_len / L))

        # padding a little. mix_len + K > pad_len >= mix_len
        # pad_len = K * L
        
        # merge s1 and s2
        s = np.stack((pad_s1,pad_s2))  # C * T
        mixtures.append(pad_mix)
        sources.append(s)

    return mixtures, sources

class AudioDataset(data.Dataset):
    
    def __init__(self, json_dir,
                 sample_rate=8000, L=int(8000*0.005)):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioDataset, self).__init__()

        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        # def sort(infos): return sorted(
        #     infos, key=lambda info: int(info[1]), reverse=True)
        # sorted_s1_infos = sort(s1_infos)
        # sorted_s2_infos = sort(s2_infos)
        self.infos = []
        for i in range(len(s1_infos)):
            self.infos.append([s1_infos[i],s2_infos[i]])

    def __getitem__(self, index):
        return self.infos[index]

    def __len__(self):
        return len(self.infos)


def _collate_fn(batch,sample_rate=8000,L=16):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x K x L, torch.Tensor
    """
    # batch should be located in list

    batch = load_mixtures_and_sources(batch,sample_rate,L)
    mixtures, sources = batch # mixtures: B * T, sources : B * C * T,T varies

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])
    
    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0,2,1)).contiguous()
    return mixtures_pad, ilens, sources_pad

class AudioDataloader(DataLoader):
    def __init__(self):
        super(AudioDataloader,self).__init__()
        self.collate_fn = _collate_fn
def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(-1) for x in xs)
    pad = xs[0].new(n_batch,* xs[0].size()[:-1], max_len, ).fill_(pad_value) #new()
    for i in range(n_batch):
        # print(pad.shape)
        pad[i, :xs[i].size(-1)] = xs[i]
    return pad
