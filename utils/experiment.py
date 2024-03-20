# The experiment class - implementing the training process itself

import torch
import numpy as np
import pandas as pd
import os
import random
import json

from utils.pl_wrapper import TrainWrapper
from utils.dataset import ForecastingDataset, ImputationDataset

from torch.utils.data import DataLoader, random_split, ConcatDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def combine_files(directory):
    # empty the combined file
    with open("./models/model_archs.py", "w") as combined_file:
        combined_file.write("")
    
    combined_content = ""
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            with open(os.path.join(directory, filename), "r") as file:
                combined_content += file.read() + "\n"

    with open("./models/model_archs.py", "w") as combined_file:
        combined_file.write(combined_content)


def load_model(model_name):
    import models.model_archs as model_archs

    if hasattr(model_archs, model_name):
        return getattr(model_archs, model_name)
    else:
        raise AttributeError(f"Model class '{model_name}' not found in the combined models file.")


class Experiment:
    def __init__(self, args):
        self.args = args
        
        # random seed
        fix_seed = args.seed

        print(f"setting random seed to {fix_seed}")
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        # model parameters
        print(f"Loading model parameters from '{args.model_config}'")
        with open(args.model_config, "r") as file:
            model_params = json.load(file)  # load the model parameters from the config file as a dictionary
        
        # model
        combine_files(args.models_path)

        print(f"Creating model '{args.model_name}' with parameters: {model_params}")
        self.model = load_model(args.model_name)(**model_params)

        # training wrapper
        print(f"Creating training wrapper with losses: {args.losses}, optimizer: {args.optimizer}, lr: {args.lr}, weight_decay: {args.weight_decay}")
        self.wrapper = TrainWrapper(
            self.model,
            args.losses,
            args.optimizer,
            args.lr,
            args.weight_decay
        )

        # data
        if args.task == "forecasting":
            print(f"Creating forecasting datasets with lookback: {args.lookback}, horizon: {args.horizon}")
            datasets = [ForecastingDataset(data_path=os.path.join(args.data_path, file),
                                           lookback=args.lookback, horizon=args.horizon)
                                           for file in os.listdir(args.data_path) if file.endswith(".csv")]

        elif args.task == "imputation":
            print(f"Creating imputation datasets with lookback: {args.lookback}, mask_prob: {args.mask_prob}")
            datasets = [ImputationDataset(data_path=os.path.join(args.data_path, file),
                                           lookback=args.lookback, mask_prob=args.mask_prob)
                                           for file in os.listdir(args.data_path) if file.endswith(".csv")]
        else:
            raise ValueError(f"Task '{args.task}' not supported")
        
        # combining datasets
        print(f"Combining {len(datasets)} datasets")
        self.dataset = ConcatDataset(datasets)

        # splitting the dataset
        print(f"Splitting the dataset with validation split: {args.val_split}")
        val_size = int(args.val_split * len(self.dataset))
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size]) ## allow for and additional predefined split
        

        # dataloaders
        print(f"Creating dataloaders with batch size: {args.batch_size}, num workers: {args.num_workers}")
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size,
                                       num_workers=args.num_workers, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size,
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
    
    def run(self):
        print("Starting the experiment...")
        print(f"Training the model '{self.args.model_name}' on the task '{self.args.task}'")
        print(f"Using the datasets from '{self.args.data_path}'")
        print(f"Using the models from '{self.args.models_path}'")
        print(f"Logging to '{self.args.logger}'")
        print(f"Saving models to '{self.args.save_dir}'")
        print(f"Training for {self.args.max_epochs} epochs")
        print()

        self.train()

        print("Experiment finished.")
