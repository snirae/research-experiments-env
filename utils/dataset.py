# The generic dataset classes for the training

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


def count_lines(file_path):
    with open(file_path, 'rb') as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        line_count = 0
        chunk_size = 4096
        read_chars = 0

        while read_chars < file_size:
            chunk = file.read(chunk_size)
            read_chars += len(chunk)
            line_count += chunk.count(b'\n')

        return line_count + 1  # Add 1 for the last line without a newline character
    

class ForecastingDataset(Dataset):
    def __init__(self, data_path, lookback, horizon, gap=0):
        self.data_path = data_path
        self.lookback = lookback
        self.horizon = horizon
        self.gap = gap
        
        self.chunksize = lookback + horizon + gap
        self.length = count_lines(data_path) - self.horizon - self.lookback - self.gap
        
        self.data = pd.read_csv(self.data_path)
        time_cols = [col for col in self.data.columns if self.data[col].dtype == pd.Timestamp]
        if time_cols:
            self.data = self.data.drop(columns=time_cols)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback].values
        y = self.data[idx+self.lookback+self.gap:idx+self.chunksize].values

        if y.shape[0] != self.horizon:
            # duplicate the last value to match the horizon
            y = np.concatenate([y, np.repeat(y[-1][None, :], self.horizon - y.shape[0], axis=0)], axis=0)

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float32)
    

class ImputationDataset(Dataset):
    def __init__(self, data_path, lookback, mask_perc):
        self.data_path = data_path
        self.lookback = lookback
        self.mask_perc = mask_perc

        self.to_mask = int(lookback * mask_perc)
        
        self.length = count_lines(data_path) - self.lookback

        self.data = pd.read_csv(self.data_path)
        time_cols = [col for col in self.data.columns if self.data[col].dtype == pd.Timestamp]
        if time_cols:
            self.data = self.data.drop(columns=time_cols)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        y = self.data[idx:idx+self.lookback].values

        # mask whole random continuous segment of the data, such that mask_perc of the data is masked
        mask_start = np.random.randint(0, self.lookback - self.to_mask)
        mask_end = mask_start + self.to_mask
        
        x = y.copy()
        x[mask_start:mask_end] = 0

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float32)
