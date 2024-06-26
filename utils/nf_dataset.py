import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.preprocessing import StandardScaler

from utils.moirai_dataset import get_pandas_dataframe


def prepare_dataset_for_nf(df, time_col, test_split=0.1, scale=True):
   # split the data
    test_size = int(len(df) * test_split)
    train_size = len(df) - test_size

    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    if scale:
        # scale the numerical columns
        scaler = StandardScaler()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
        test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])
    else:
        scaler = None

    # convert to neuralforecast format - long dataset format with 3 columns: unique_id, ds, y
    # unique_id is the index of the time series, ds is the timestamp, y is the value
    train_dfs = []
    test_dfs = []
    for col in df.columns:
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

    return train_data, test_data, scaler

def load_dataset_for_nf(data_path, time_col=None, test_split=0.1, scale=True,
                        is_local=True):
    if is_local:
        file_type = data_path.split(".")[-1]
        data = load_dataset(file_type, data_files=data_path)['train']
        data = data.to_pandas()
    else:
        data = get_pandas_dataframe(data_path)
        time_col = 'date'
    
    train_data, test_data, scaler = prepare_dataset_for_nf(data, time_col, test_split, scale)

    return train_data, test_data, scaler
