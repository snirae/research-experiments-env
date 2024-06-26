import numpy as np
import pandas as pd
from collections.abc import Generator
from typing import Any
from pathlib import Path

import datasets
from gluonts.dataset.pandas import PandasDataset
from datasets import Features, Sequence, Value, load_dataset
from sklearn.preprocessing import StandardScaler

from uni2ts.data.builder.simple import SimpleDatasetBuilder


def splitter(df, val_split=0.1, test_split=0.1):
   """
   Split the data into train, validation, and test sets
   """
   n = len(df)
   val_size = int(n * val_split)
   test_size = int(n * test_split)

   train = df.iloc[:-(val_size + test_size)]
   val = df.iloc[-(val_size + test_size):-test_size]
   test = df.iloc[-test_size:]

   return train, val, test

def generate_generator(df):
   def multivar_example_gen_func() -> Generator[dict[str, Any], None, None]:
      yield {
            "target": df.to_numpy().T,  # array of shape (var, time)
            "start": df.index[0],
            "freq": pd.infer_freq(df.index),
            "item_id": "item_0",
      }

   return multivar_example_gen_func

def remove_nan(df: pd.DataFrame):
   nan_percentages = df.isnull().mean() * 100

   if nan_percentages.max() > 80:
      print("High NaN percentage detected. Applying aggressive strategy.")
      strategy = "aggressive"
   elif nan_percentages.mean() > 30:
      print("Moderate NaN percentage detected. Applying moderate strategy.")
      strategy = "moderate"
   else:
      print("Low NaN percentage detected. Applying conservative strategy.")
      strategy = "conservative"

   if strategy == "aggressive":
      columns_to_drop = nan_percentages[nan_percentages > 50].index
      df = df.drop(columns=columns_to_drop)
      print(f"Dropped {len(columns_to_drop)} columns with >50% NaNs")

      df = df.interpolate(method='time', limit_direction='both')
      df = df.fillna(method='ffill').fillna(method='bfill')

   elif strategy == "moderate":
      columns_to_drop = nan_percentages[nan_percentages > 70].index
      df = df.drop(columns=columns_to_drop)
      print(f"Dropped {len(columns_to_drop)} columns with >70% NaNs")

      df = df.interpolate(method='time', limit_direction='both')
      df = df.fillna(df.mean())

   else: 
      df = df.interpolate(method='time', limit_direction='both')
      df = df.fillna(method='ffill').fillna(method='bfill')

   return df

def df_to_hfs(df, val_split=0.1, test_split=0.1, scale=True):
   if scale:
      # scale the numerical columns
      scaler = StandardScaler()
      numerical_cols = df.select_dtypes(include=[np.number]).columns
      df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

   train, val, test = splitter(df, val_split, test_split)

   train_gen = generate_generator(train)
   val_gen = generate_generator(val)

   features = Features(
      dict(
         target=Sequence(
            Sequence(Value("float32")), length=len(df.columns)
            ),  # multivariate time series are saved as (var, time)
            start=Value("timestamp[s]"),
            freq=Value("string"),
            item_id=Value("string"),
            )
   )

   train_dataset = datasets.Dataset.from_generator(
      train_gen, features=features
   )
   val_dataset = datasets.Dataset.from_generator(
      val_gen, features=features
   )

   return train_dataset, val_dataset, test

def prepare_dataset_for_moirai(data_path, time_col, scale=True):
   file_type = data_path.split(".")[-1]
   data = load_dataset(file_type, data_files=data_path)['train']
   df = data.to_pandas()   
   df = df.set_index(time_col)
   df.index = pd.to_datetime(df.index)

   df = remove_nan(df)

   return df_to_hfs(df, scale=scale)

def get_dataset(dataset_name): # downloads specific dataset in GluonTS format
    dataset_path = "Salesforce/lotsa_data"
    dataset = load_dataset(dataset_path, dataset_name)    

    data_freq = dataset['train']['freq'][0]
    data = dataset['train'].to_pandas()
    data = pd.DataFrame(data)

    dataset = PandasDataset.from_long_dataframe(
      dataframe=data,
      item_id="item_id",
      timestamp="start",
      target="target",
      freq=data_freq
    )
    return dataset

def flatten_y(y):
    flattened = []
    for element in y:
        if isinstance(element, np.ndarray):
            flattened.extend(flatten_y(element))
        else:
            flattened.append(element)
    return np.array(flattened)

def get_pandas_dataframe(dataset_name):  # returns dataframe with normal structure
    dataset = get_dataset(dataset_name)
    
    data_dict = {}
    all_dates = pd.Index([])

    for item in dataset:
        item_id = item['item_id']
        start_date = item['start'].to_timestamp()
        target_values = flatten_y(item['target'])
        dates = pd.date_range(start=start_date, periods=len(target_values), freq=dataset.freq)

        if item_id not in data_dict:
            data_dict[item_id] = pd.Series(target_values, index=dates)
        else:
            data_dict[item_id] = pd.concat([data_dict[item_id], pd.Series(target_values, index=dates)])

        all_dates = all_dates.union(dates)

    for item_id in data_dict:
        data_dict[item_id] = data_dict[item_id].reindex(all_dates)

    df = pd.DataFrame(data_dict)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)

    return df

def load_dataset_for_moirai(data_path, time_col, transform_map,
                            val_split=0.1, test_split=0.1,
                            horizon=96, scale=True, is_local=True):
   if is_local:
      hf_dataset = prepare_dataset_for_moirai(data_path, time_col=time_col, scale=scale)
      train, val, test = hf_dataset

      path = data_path.split("/")[:-1]
      path = "/".join(path)

      name = data_path.split("/")[-1]
      name = name.split(".")[0] if "." in name else name

      path += f"/{name}"
   else:
      df = get_pandas_dataframe(data_path)
      df = df.set_index('date')
      df.index = pd.to_datetime(df.index)

      df = remove_nan(df)
      
      train, val, test = df_to_hfs(df, val_split=val_split, test_split=test_split, scale=scale)

      path = f'../data/{data_path}'
      name = data_path

   # save train, val to disk
   train.save_to_disk(f"{path}/{name}_train")
   val.save_to_disk(f"{path}/{name}_val")

   # uni2ts TimeSeriesDataset format for train and val
   SimpleDatasetBuilder.__post_init__ = lambda self: setattr(self, 'storage_path', Path(path))

   train_dataset = SimpleDatasetBuilder(dataset=f'{name}_train').load_dataset(transform_map)
   val_dataset = SimpleDatasetBuilder(dataset=f'{name}_val').load_dataset(transform_map)
   
   return train_dataset, val_dataset, test
