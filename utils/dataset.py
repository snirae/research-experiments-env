# The generic dataset classes for the training

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import subprocess


def count_lines(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE)
    output = result.stdout.decode().strip()
    num_lines = int(output.split()[0])

    return num_lines
    

class ForecastingDataset(Dataset):
    def __init__(self, data_path, lookback, horizon):
        self.data_path = data_path
        self.lookback = lookback
        self.horizon = horizon
        
        self.chunksize = lookback + horizon
        self.length = count_lines(data_path) - self.horizon - self.lookback

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk = pd.read_csv(self.data_path, skiprows=idx, nrows=self.chunksize)

        # remove the date column if it exists
        time_cols = [col for col in chunk.columns if chunk[col].dtype == pd.Timestamp]
        if time_cols:
            chunk = chunk.drop(columns=time_cols)

        x = chunk[:self.lookback].values
        y = chunk[self.lookback:].values

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float32)
    

class ImputationDataset(Dataset):
    def __init__(self, data_path, lookback, mask_prob):
        self.data_path = data_path
        self.lookback = lookback
        self.mask_prob = mask_prob
        
        self.length = count_lines(data_path) - self.lookback

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        chunk = pd.read_csv(self.data_path, skiprows=idx, nrows=self.lookback)

        # remove the date column if it exists
        time_cols = [col for col in chunk.columns if chunk[col].dtype == pd.Timestamp]
        if time_cols:
            chunk = chunk.drop(columns=time_cols)

        # mask the data with the given probability
        mask = np.random.binomial(1, 1 - self.mask_prob, chunk.shape)

        y = chunk.values
        x = y * mask

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float32)
