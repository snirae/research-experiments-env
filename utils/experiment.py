# The experiment class - implementing the training process itself

import torch
import numpy as np
import pandas as pd
import os
import random
import json
import matplotlib.pyplot as plt

from utils.pl_wrapper import TrainWrapper
from utils.dataset import ForecastingDataset, ImputationDataset

from testing.plotting import plot_runs_comparison

from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from models.nf_models import load_model
from utils.lora import lora_finetune
from huggingface_hub import hf_hub_download
from uni2ts.model.moirai import MoiraiForecast



class Experiment:
    def __init__(self, args):
        self.args = args
        
        fix_seed = args.seed # random seed

        print(f"setting random seed to {fix_seed}")
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        # model parameters
        print(f"Loading model parameters from '{args.model_config}'")
        with open(args.model_config, "r") as file:
            model_params = json.load(file)  # load the model parameters from the config file as a dictionary
        
        
        # model
        if args.model_name.lower() == "moirai":
            size = "small"
            model = MoiraiForecast.load_from_checkpoint(
                checkpoint_path=hf_hub_download(
                    repo_id=f"Salesforce/moirai-1.0-R-{size}", filename="model.ckpt"
                ),
                prediction_length=model_params['prediction_length'],
                context_length=model_params['context_length'],
                patch_size=model_params['patch_size'],
                num_samples=1,
                target_dim=2,
                # feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                # past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
                # map_location="cuda:0" if torch.cuda.is_available() else "cpu",
                )
            
            self.model = lora_finetune(model)

            print(self.model)
            
            raise NotImplementedError("Moirai model is not yet supported")

        else:
            model_class = load_model(args.model_name)

            print(f"Creating model '{args.model_name}' with parameters: {model_params}")
            self.model = model_class(**model_params)

        # training wrapper
        print(f"Creating training wrapper with losses: {args.losses}, optimizer: {args.optimizer}, lr: {args.lr}, weight_decay: {args.weight_decay}")
        self.wrapper = TrainWrapper(
            self.model,
            args.losses,
            args.optimizer,
            args.lr,
            args.weight_decay
        )

        # loading checkpoint
        if args.ckpt_path:
            print(f"Loading model from checkpoint '{args.ckpt_path}'")
            self.wrapper.load_from_checkpoint(args.ckpt_path)

        # data
        # if args.task == "forecasting":
        #     print(f"Creating forecasting datasets with lookback: {args.lookback}, horizon: {args.horizon}")
        #     datasets = [ForecastingDataset(data_path=os.path.join(args.data_path, file),
        #                                    lookback=args.lookback, horizon=args.horizon,
        #                                    gap=args.gap)
        #                                    for file in os.listdir(args.data_path) if file.endswith(".csv")]

        # elif args.task == "imputation":
        #     print(f"Creating imputation datasets with lookback: {args.lookback}, mask_perc: {args.mask_perc}")
        #     datasets = [ImputationDataset(data_path=os.path.join(args.data_path, file),
        #                                   lookback=args.lookback, mask_perc=args.mask_perc)
        #                                   for file in os.listdir(args.data_path) if file.endswith(".csv")]
        # else:
        #     raise ValueError(f"Task '{args.task}' not supported")
        
        # splitting the datasets
        # print(f"Splitting the datasets with validation split: {args.val_split}")

        # trains, vals = [], []
        # for dataset in datasets:
        #     val_size = int(args.val_split * len(dataset))
        #     train_size = len(dataset) - val_size

        #     val_size = int(args.val_split * len(dataset))
        #     train_size = len(dataset) - val_size

        #     train_idxs = list(range(train_size))
        #     val_idxs = list(range(train_size, len(dataset)))

        #     train = Subset(dataset, train_idxs)
        #     val = Subset(dataset, val_idxs)

        #     trains.append(train)
        #     vals.append(val)

        # # combining datasets
        # print(f"Combining {len(datasets)} datasets")
        # self.train_dataset = ConcatDataset(trains)
        # self.val_dataset = ConcatDataset(vals)

        # print(f"Total length of the datasets: {len(self.train_dataset) + len(self.val_dataset)}")
        
        # dataset from file
        
        # dataloaders
        print(f"Creating dataloaders with batch size: {args.batch_size}, num workers: {args.num_workers}")
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size,
                                       num_workers=args.num_workers, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size,
                                     num_workers=args.num_workers, shuffle=False)
        
        # test data
        if args.test:
            if args.task == "forecasting":
                print(f"Creating test datasets with lookback: {args.lookback}, horizon: {args.horizon}")
                test_datasets = [ForecastingDataset(data_path=os.path.join(args.test_data, file),
                                                    lookback=args.lookback, horizon=args.horizon,
                                                    gap=args.gap)
                                                    for file in os.listdir(args.test_data) if file.endswith(".csv")]
            elif args.task == "imputation":
                print(f"Creating test datasets with lookback: {args.lookback}, mask_perc: {args.mask_perc}")
                test_datasets = [ImputationDataset(data_path=os.path.join(args.test_data, file),
                                                  lookback=args.lookback, mask_perc=args.mask_perc)
                                                  for file in os.listdir(args.test_data) if file.endswith(".csv")]
            
            print(f"Combining {len(test_datasets)} test datasets")
            self.test_dataset = ConcatDataset(test_datasets)
            
            print(f"Total length of the test dataset: {len(self.test_dataset)}")
            
            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size,
                                         num_workers=args.num_workers, shuffle=False)

        # callbacks
        print(f"Creating callbacks with early stopping: {args.early_stopping}, patience: {args.patience}, min improvement: {args.min_improvement}")
        callbacks = []
        if args.early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience,
                                           min_delta=args.min_improvement))
            
        callbacks.append(ModelCheckpoint(dirpath=args.save_dir, monitor="val_loss",
                                         save_top_k=1, mode="min", save_last=True))
        
        # logger
        print(f"Creating logger '{args.logger}'")
        if args.logger == "wandb":
            with open(args.api_key_file, "r") as file:
                api_key = file.read().strip()
            
            logger = WandbLogger(project=args.project, entity=args.entity, api_key=api_key)
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger(args.log_dir)
        else:
            raise ValueError(f"Logger '{args.logger}' not supported")
        
        # logging hyperparameters
        print("Logging hyperparameters")
        logger.log_hyperparams(vars(args))

        # trainer
        self.trainer = Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=args.max_epochs,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=args.log_interval
        )

    def train(self):
        self.trainer.fit(self.wrapper, self.train_loader, self.val_loader)


    def test(self):
        self.trainer.test(self.wrapper, self.test_loader)

        LOOKBACKS = 5

        forecast = np.array(self.wrapper.test_predictions)
        forecast = forecast[range(0, min(len(forecast), self.args.lookback*LOOKBACKS), self.args.lookback)]  # (num_samples, lookback, features))]
        forecast = forecast.reshape(-1, forecast.shape[-1])

        targets = np.array(self.wrapper.test_targets)
        targets = targets[range(0, min(len(targets), self.args.lookback*LOOKBACKS), self.args.lookback)]  # (num_samples, lookback, features)
        targets = targets.reshape(-1, targets.shape[-1])
        
        # plotting
        if self.args.save_plots:
            print("Saving the forecasting plots...")

            if not os.path.exists(f'{self.args.log_dir}/plots'):
                os.makedirs(f'{self.args.log_dir}/plots')
                
            plot_runs_comparison([{"label": targets, "forecast": forecast}], lookback_size=self.args.lookback,
                                 save_dir=f'{self.args.log_dir}/plots', name=self.args.model_name)
        else:
            print("Plotting the forecasting results...")
            plot_runs_comparison([{"label": targets, "forecast": forecast}], lookback_size=self.args.lookback)

    def run(self):
        print("\n\nStarting the experiment...")
        print(f"Training the model '{self.args.model_name}' on the task '{self.args.task}'")
        print(f"Using the datasets from '{self.args.data_path}'")
        print(f"Logging to '{self.args.logger}'")

        if self.args.train:
            print(f"Saving models to '{self.args.save_dir}'")
            print(f"Training for {self.args.max_epochs} epochs")
            print()

            self.train()

            print("Training finished.")

        if self.args.test:
            print("\n\nTesting the model...")
            
            self.test()

            print("Testing finished.")

        print("\nExperiment finished.")
