# The generic dataset classes for the training

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

import datasets


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


def load_dataset_for_nf(data_path, time_col, test_split=0.1):
    file_type = data_path.split(".")[-1]
    data = datasets.load_dataset(file_type, data_files=data_path)['train']
    data = data.to_pandas()

    # split the data
    test_size = int(len(data) * test_split)
    train_size = len(data) - test_size

    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    # convert to neuralforecast format - long dataset format with 3 columns: unique_id, ds, y
    # unique_id is the index of the time series, ds is the timestamp, y is the value
    train_dfs = []
    test_dfs = []
    for col in data.columns:
        if col == time_col:
            continue

        train_df = train_data[[time_col, col]].copy()
        test_df = test_data[[time_col, col]].copy()
        
        train_df.columns = ["ds", "y"]
        test_df.columns = ["ds", "y"]

        train_df["unique_id"] = col
        test_df["unique_id"] = col

        train_dfs.append(train_df)
        test_dfs.append(test_df)

    train_data = pd.concat(train_dfs)
    test_data = pd.concat(test_dfs)
    
    train_data['ds'] = pd.to_datetime(train_data['ds'])
    test_data['ds'] = pd.to_datetime(test_data['ds'])

    train_data.sort_values(["unique_id", "ds"], inplace=True)
    test_data.sort_values(["unique_id", "ds"], inplace=True)

    # reorder columns
    train_data = train_data[["unique_id", "ds", "y"]]
    test_data = test_data[["unique_id", "ds", "y"]]

    return train_data, test_data
