import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    names, lengths, dv3s, e_labs, g_labs, s_labs = zip(*batch)

    dv3_list = []
    for f in dv3s:
        f_tensor = torch.from_numpy(f)
        if f_tensor.ndim == 3:
            f_tensor = f_tensor[:, 1, :]
        dv3_list.append(f_tensor)
    dv3_padded = pad_sequence(dv3_list, batch_first=True)

    e_padded = pad_sequence([torch.from_numpy(l) for l in e_labs], batch_first=True, padding_value=-100)
    g_padded = pad_sequence([torch.from_numpy(l) for l in g_labs], batch_first=True, padding_value=-100)

    s_labs = torch.tensor(s_labs).float()
    lengths = torch.tensor(lengths).int()

    return names, lengths, dv3_padded, e_padded, g_padded, s_labs


class CustomVideoDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.pkl_dir = os.path.join(os.path.dirname(root_dir), 'pkl_files')

        csv_name = 'train.csv' if train else 'test.csv'
        self.video_folders = pd.read_csv(os.path.join(root_dir, csv_name), header=None)[0].tolist()

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_name = self.video_folders[idx]
        video_path = os.path.join(self.pkl_dir, video_name.replace('.csv', '.pkl'))

        with open(video_path, 'rb') as f:
            video_data = pickle.load(f)

        dv3_features = video_data['dino_v3_b224_feature'].astype('float32')
        e_labels = video_data['error_GT'].astype('float32')
        g_labels = video_data['gesture_GT'].astype('float32')
        s_label = video_data['GRS_GT'][0].astype('float32')
        video_length = len(g_labels)

        return video_name, video_length, dv3_features, e_labels, g_labels, s_label
