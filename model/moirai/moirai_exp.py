import json
import numpy as np
import pandas as pd
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from utils.experiment import Experiment
from model.moirai.moirai import MoiraiHandler
from utils.moirai_dataset import load_dataset_for_moirai


class MoiraiExp(Experiment):
    def __init__(self, args, i):
        super(MoiraiExp, self).__init__(args)

        # callbacks
        print(f"Creating callbacks with early stopping: {args.early_stopping}, patience: {args.patience}, min improvement: {args.min_improvement}")
        es = EarlyStopping(
            monitor='val/PackedNLLLoss',
            patience=args.patience,
            min_delta=args.min_improvement,
            mode='min'
        )
        mc = ModelCheckpoint(
            monitor='val/PackedNLLLoss',
            filename='moirai' + '-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )

        self.callbacks = [es, mc]

        # model parameters
        print(f"Loading model parameters from '{args.configs[i]}'")
        with open(args.configs[i], "r") as file:
            params = json.load(file)

        self.params = params

        # model
        print(f"Loading model: moirai")
        self.moirai = MoiraiHandler(
            args,
            size=params['size'],
            horizon=params['horizon'],
            lookback=params['lookback'],
            patch_size=params['patch_size'],
            num_samples=params['num_samples'],
            target_dim=params['target_dim'],
            lora=params['lora']
        )

        # data
        print(f"Loading data from '{args.data_path}'")
        train_set, val_set, test_df = load_dataset_for_moirai(
            args.data_path,
            time_col=args.time_col,
            transform_map=self.moirai.train_transform_map,
            val_split=args.val_split,
            test_split=args.test_split,
            horizon=args.horizon,
            scale=args.norm,
            is_local=args.is_local
        )
        
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_df

        # logger
        print(f"Creating logger: {args.logger}")
        if args.logger == "wandb":
            with open(args.api_key_file, "r") as file:
                api_key = file.read().strip()
            wandb.login(key=api_key)
            
            logger = WandbLogger(save_dir=args.log_dir,
                                 project=args.project if args.project else f'{args.dataset_name}_{args.horizon}',
                                 entity=args.entity,
                                 name=f'moirai_{args.dataset_name}_{args.horizon}')
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger(args.log_dir,
                                       name=f'{args.dataset_name}_{args.horizon}',
                                       version=f'moirai_{args.dataset_name}_{args.horizon}')
        else:
            raise ValueError(f"Logger '{args.logger}' not supported")
        
        self.logger = logger
        logger.log_hyperparams(args.__dict__)

    def train(self):
        trainer = pl.Trainer(
            logger=self.logger,
            callbacks=self.callbacks,
            max_steps=self.args.max_steps,
            accelerator=self.args.accelerator,
            log_every_n_steps=self.args.log_interval,
        )
        self.trainer = trainer

        self.moirai.train(trainer, self.train_set, self.val_set, self.params)

    def test(self):
        labels, forecasts = self.moirai.predict(self.test_set)
        
        mse = np.mean((labels - forecasts) ** 2)
        mae = np.mean(np.abs(labels - forecasts))

        print(f"MOIRAI - MSE: {mse:.4f}, MAE: {mae:.4f}")

        # logging
        self.logger.log_metrics({'MSE': mse, 'MAE': mae})