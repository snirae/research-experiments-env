# The experiment class - implementing the training process itself

import torch
import numpy as np
import pandas as pd
import os
import random
import json
import matplotlib.pyplot as plt

from utils.dataset import load_dataset_for_nf

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb

from model.nf_models import load_model
from utils.training import load_loss, load_optimizer

from utils.nf_fit import wrapper, predict
from neuralforecast import NeuralForecast
from neuralforecast.common._base_windows import BaseWindows
from statsforecast import StatsForecast

import logging


class Experiment:
    def __init__(self, args):
        args.dataset_name = args.data_path.split("/")[-1].split(".")[0]
        self.args = args

        NeuralForecast.fit = wrapper(args=args)
        BaseWindows.predict = predict
        
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
        self.train_df, self.test_df, self.scaler = load_dataset_for_nf(args.data_path, args.time_col, scale=args.norm)
        freq = pd.infer_freq(self.train_df['ds'].unique())  # infer frequency from the data

        # callbacks
        print(f"Creating callbacks with early stopping: {args.early_stopping}, patience: {args.patience}, min improvement: {args.min_improvement}")
        callbacks = []
        for model_name in args.models:
            if args.early_stopping:
                e = EarlyStopping(
                        monitor='valid_loss',
                        patience=args.patience,
                        min_delta=args.min_improvement,
                        mode='min'
                    )
            mc = ModelCheckpoint(
                    monitor='valid_loss',
                    filename=f'{model_name}' + '-{epoch:02d}-{valid_loss:.2f}',
                    save_top_k=1,
                    mode='min',
                )
            callbacks.append([e, mc])
        
        # model parameters
        print(f"Loading models parameters from '{args.configs}'")
        configs = []
        for config in args.configs:
            with open(config, "r") as file:
                configs.append(json.load(file))
        
        # models
        models_classes = [load_model(model_name) for model_name in args.models]

        self.models = []
        for i, model_class in enumerate(models_classes):
            model_params = configs[i]
            print(f"Creating model '{args.models[i]}' with parameters: {model_params}")
            model = model_class(**model_params,
                                
                                # BaseModel kwargs
                                val_check_steps=args.val_interval,
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
                                callbacks=callbacks[i],
                                log_every_n_steps=args.log_interval,
                                enable_checkpointing=True)

            self.models.append(model)

        self.nf = NeuralForecast(
            models=self.models,
            freq=freq,
        )

    def test_predict(self, df):
        # split to lookback and forecast windows
        dates_split = df['ds'].unique()[list(range(self.args.lookback, len(df['ds'].unique()), self.args.horizon))]
        forecasts = {model_name: [] for model_name in self.args.models}
        gts = []

        for i in range(len(dates_split) - 1):
            lookback_df = df[df['ds'] < dates_split[i]]
            forecast_df = df[(df['ds'] >= dates_split[i]) & (df['ds'] < dates_split[i + 1])]

            forecast = self.nf.predict(df=lookback_df)
            for model_name in self.args.models:
                forecasts[model_name].append(forecast[model_name].values)
            gts.append(forecast_df['y'].values)

        return forecasts, np.array(gts)

    def train(self):
        val_size = int(len(self.train_df['ds'].unique()) * self.args.val_split)
        self.nf.fit(df=self.train_df, val_size=val_size)

    def test(self):
        # disable logging of pytorch_lightning
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # test logger
        if self.args.logger == "wandb":
            logger = WandbLogger(save_dir=self.args.log_dir,
                                 project=self.args.project if self.args.project else self.args.dataset_name,
                                 entity=self.args.entity,
                                 name="test-results")
        elif self.args.logger == "tensorboard":
            logger = TensorBoardLogger(self.args.log_dir,
                                       name=self.args.dataset_name,
                                       version="test-results")
            
        logger.log_hyperparams(vars(self.args))

        y_hat_dict, y = self.test_predict(self.test_df)
        print(f"Predicted {len(y)} windows\n\n")

        mses, maes = [], []
        for i, model_name in enumerate(self.args.models):
            y_hat = np.array(y_hat_dict[model_name])

            # metrics
            mae = np.mean(np.abs(y - y_hat))
            mse = np.mean((y - y_hat) ** 2)

            print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}")

            mses.append(mse)
            maes.append(mae)

        # logging
        if self.args.logger == "wandb":
            df = pd.DataFrame({
                "model": self.args.models,
                "mse": mses,
                "mae": maes
            })
            table = wandb.Table(dataframe=df)
            logger.experiment.log({"test_metrics": table})
        elif self.args.logger == "tensorboard":
            for i, model_name in enumerate(self.args.models):
                logger.log_metrics({f"{model_name}_mse": mses[i],
                                    f"{model_name}_mae": maes[i]})

        # # plotting
        # StatsForecast.plot(df=self.test_df,
        #                    forecasts_df=tr,
        #                    models=[self.args.model_name],
        #                    engine='matplotlib',)

    def run(self):
        print("\n\nStarting the experiment...")
        print(f"Training the models {self.args.models}")

        if self.args.train:
            print(f"Saving checkpoints to '{self.args.save_dir}'")
            print(f"Training for {self.args.max_steps} steps")
            print()

            self.train()

            print("Training finished.")

        if self.args.test:
            print("\n\nTesting the model...")
            
            self.test()

            print("\nTesting finished.")

        print("\nExperiment finished.")

        if self.args.logger == "wandb":
            wandb.finish()
