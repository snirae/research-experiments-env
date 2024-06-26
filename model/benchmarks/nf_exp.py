# Neuralforecast experiment

import pandas as pd
import json
import numpy as np
import logging
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils.experiment import Experiment
from utils.nf_dataset import load_dataset_for_nf
from model.benchmarks.training import load_loss
from model.benchmarks.neuralmodels import NFHandler


class NFExp(Experiment):
    def __init__(self, args):
        super(NFExp, self).__init__(args)

        # loss function
        print(f"Loading loss function: {args.loss}")
        loss = load_loss(args.loss)()

        self.loss = loss

        # data
        print(f"Loading data from '{args.data_path}'")
        self.train_df, self.test_df, self.scaler = load_dataset_for_nf(args.data_path, args.time_col, args.test_split, args.norm)
        freq = pd.infer_freq(self.train_df['ds'].unique())

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

        self.callbacks = callbacks

        # model parameters
        print(f"Loading models parameters from '{args.configs}'")
        configs = []
        for config in args.configs:
            with open(config, "r") as file:
                configs.append(json.load(file))

        self.configs = configs

        # models
        print(f"Loading models: {args.models}")
        self.nf = NFHandler(
            models=args.models,
            params=configs,
            config=args,
            freq=freq,
            loss=loss,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            callbacks=callbacks
        )

    def train(self):
        val_size = int(len(self.train_df['ds'].unique()) * self.args.val_split)
        self.nf.train(self.train_df, val_size)

    def test(self):
        # disable logging of pytorch_lightning
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # test logger
        if self.args.logger == "wandb":
            logger = WandbLogger(save_dir=self.args.log_dir,
                                 project=self.args.project if self.args.project else f'{self.args.dataset_name}_{self.args.horizon}',
                                 entity=self.args.entity,
                                 name="test-results")
        elif self.args.logger == "tensorboard":
            logger = TensorBoardLogger(self.args.log_dir,
                                       name=f'{self.args.dataset_name}_{self.args.horizon}',
                                       version="test-results")
            
        logger.log_hyperparams(vars(self.args))

        y_hat_dict, y = self.nf.test_predict(self.test_df, self.args.lookback, self.args.horizon)
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
