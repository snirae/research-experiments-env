# The experiment class - implementing the training process itself

import torch
import numpy as np
import pandas as pd
import os
import random
import json
import matplotlib.pyplot as plt

from utils.pl_wrapper import TrainWrapper
from utils.gluon_dataset import get_dataset
from gluonts.dataset.split import split

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
            )

            self.model = lora_finetune(model)
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
        
        # dataset
        dataset = get_dataset(source="huggingface",
                              dataset_path="Salesforce/lotsa_data",
                              dataset_name='bull')
        
        # Split into train/test set
        PDT = args.horizon + args.gap  # prediction length
        TEST = int(len(dataset) * args.test_split)
        train, test_template = split(
            dataset, offset=-TEST
        )  # assign last TEST time steps as test set

        # Split into train/validation set
        VAL = int(len(train) * args.val_split)
        train, val = split(
            train, offset=-VAL
        )

        # Construct rolling window evaluation
        test_data = test_template.generate_instances(
            prediction_length=PDT,  # number of time steps for each prediction
            windows=TEST // PDT,  # number of windows in rolling window evaluation
            distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
        )

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
