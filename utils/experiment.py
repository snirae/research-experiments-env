# The experiment class - implementing the training process itself

import torch
import numpy as np
import pandas as pd
import os
import random
import json
import matplotlib.pyplot as plt

from utils.dataset import load_dataset_for_nf
from testing.plotting import plot_runs_comparison

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model.nf_models import load_model
from utils.training import load_loss, load_optimizer

from neuralforecast import NeuralForecast
from statsforecast import StatsForecast


class Experiment:
    def __init__(self, args):
        self.args = args
        
        # random seed
        fix_seed = args.seed

        print(f"setting random seed to {fix_seed}")
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        # loss function
        print(f"Loading loss function: {args.losses}")
        # loss_functions = [load_loss(loss)() for loss in args.losses]
        # loss = LossSum(loss_functions)
        loss = load_loss(args.losses[0])()

        # optimizer
        print(f"Loading optimizer '{args.optimizer}'")
        optimizer = load_optimizer(args.optimizer)
        optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}

        # data
        print(f"Loading data from '{args.data_path}'")
        self.train_df, self.test_df = load_dataset_for_nf(args.data_path, args.time_col)
        freq = pd.infer_freq(self.train_df['ds'].unique())  # infer frequency from the data

        # callbacks
        print(f"Creating callbacks with early stopping: {args.early_stopping}, patience: {args.patience}, min improvement: {args.min_improvement}")
        callbacks = []
        if args.early_stopping:
            callbacks.append(EarlyStopping(monitor="valid_loss", patience=args.patience,
                                           min_delta=args.min_improvement))
            
        callbacks.append(ModelCheckpoint(dirpath=args.save_dir, monitor="valid_loss",
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

        # model parameters
        print(f"Loading model parameters from '{args.model_config}'")
        with open(args.model_config, "r") as file:
            model_params = json.load(file)  # load the model parameters from the config file as a dictionary
        
        # model
        model_class = load_model(args.model_name)

        print(f"Creating model '{args.model_name}' with parameters: {model_params}")
        self.model = model_class(**model_params,
                                 
                                 # BaseModel kwargs
                                 val_check_steps=10,
                                 early_stop_patience_steps=args.patience,
                                 random_seed=fix_seed,
                                 loss=loss,
                                 optimizer=optimizer,
                                 optimizer_kwargs=optimizer_kwargs,

                                 batch_size=args.batch_size,
                                 max_steps=args.max_steps,
                                 
                                 # trainer kwargs
                                 accelerator=args.accelerator,
                                 devices=args.devices,
                                 logger=logger,
                                 callbacks=callbacks,
                                 log_every_n_steps=args.log_interval,
                                 enable_checkpointing=True)

        # loading checkpoint
        if args.ckpt_path:
            print(f"Loading model from checkpoint '{args.ckpt_path}'")
            self.model.load_from_checkpoint(args.ckpt_path)

        self.nf = NeuralForecast(
            models=[self.model],
            freq=freq,
        )

    def train(self):
        val_size = int(len(self.train_df['ds'].unique()) * self.args.val_split)
        self.nf.fit(df=self.train_df, val_size=val_size)

    def test(self):
        self.test_results = tr = self.nf.predict(df=self.test_df)

        y_hat = tr[self.args.model_name].values.reshape(self.args.horizon, -1)
        y = self.test_df['y'].values.reshape(-1, y_hat.shape[1])[-self.args.horizon:]

        # metrics
        mae = np.mean(np.abs(y - y_hat))
        mse = np.mean((y - y_hat) ** 2)
        mape = np.mean(np.abs(y - y_hat) / y)
        smape = np.mean(2 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))

        print(f"MAE: {mae}, MSE: {mse}, MAPE: {mape}, SMAPE: {smape}")

        # plotting
        StatsForecast.plot(df=self.test_df,
                           forecasts_df=tr,
                           models=[self.args.model_name],
                           engine='matplotlib',)

    def run(self):
        print("\n\nStarting the experiment...")
        print(f"Training the model '{self.args.model_name}'")

        if self.args.train:
            print(f"Saving checkpoints to '{self.args.save_dir}'")
            print(f"Training for {self.args.max_steps} steps")
            print()

            self.train()

            print("Training finished.")

        if self.args.test:
            print("\n\nTesting the model...")
            
            self.test()

            print("Testing finished.")

        print("\nExperiment finished.")
