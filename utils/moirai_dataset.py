import numpy as np
import pandas as pd
from collections.abc import Generator
from typing import Any
from pathlib import Path

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split

import datasets
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

   return df_to_hfs(df, scale=scale)

def load_dataset_for_moirai(data_path, time_col, transform_map,
                            val_split=0.1, test_split=0.1,
                            horizon=96, scale=True, is_local=True):
   if is_local:
      hf_dataset = prepare_dataset_for_moirai(data_path, time_col=time_col, scale=scale)
      train, val, test = hf_dataset
   else:
      hf_dataset = load_dataset(data_path)['train']
      df = hf_dataset.to_pandas()
      df = df.set_index(time_col)
      train, val, test = df_to_hfs(df, val_split=val_split, test_split=test_split, scale=scale)

   # save train, val to disk
   path = data_path.split("/")[:-1]
   path = "/".join(path)

   name = data_path.split("/")[-1]
   name = name.split(".")[0] if "." in name else name

   train.save_to_disk(f"{path}/{name}_train")
   val.save_to_disk(f"{path}/{name}_val")

   # uni2ts TimeSeriesDataset format for train and val
   SimpleDatasetBuilder.__post_init__ = lambda self: setattr(self, 'storage_path', Path('/'.join(path)))

   train_dataset = SimpleDatasetBuilder(dataset=f'{name}_train').load_dataset(transform_map)
   val_dataset = SimpleDatasetBuilder(dataset=f'{name}_val').load_dataset(transform_map)
   
   return train_dataset, val_dataset, test
