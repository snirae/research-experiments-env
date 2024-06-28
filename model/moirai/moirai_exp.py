import yaml
import numpy as np
import wandb
import torch

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from utils.experiment import Experiment
from model.moirai.moirai import MoiraiHandler
from utils.moirai_dataset import load_dataset_for_moirai


class MoiraiExp(Experiment):
    def __init__(self, args, i):
        super(MoiraiExp, self).__init__(args)

        # callbacks
        print(f"Creating callbacks with early stopping: {args.early_stopping}, patience: {args.patience}, min improvement: {args.min_improvement}")
        es = EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            min_delta=args.min_improvement,
            mode='min',
            verbose=True
        )
        mc = ModelCheckpoint(
            dirpath=f'args.save_dir/{args.dataset_name}_{args.horizon}/moirai',
            monitor='val_loss',
            filename='moirai' + '-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min'
        )

        self.callbacks = [es, mc]

        # model parameters
        print(f"Loading model parameters from '{args.configs[i]}'")
        with open(args.configs[i], "r") as file:
            params = yaml.safe_load(file)

        self.params = params

        # model
        print(f"Loading model: moirai")
        self.moirai = MoiraiHandler(
            args,
            size=params['size'],
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
                                 name=f'moirai_{params["size"]}')
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger(args.log_dir,
                                       name=f'{args.dataset_name}_{args.horizon}',
                                       version=f'moirai_{params["size"]}')
        else:
            raise ValueError(f"Logger '{args.logger}' not supported")
        
        self.logger = logger
        logger.log_hyperparams(args.__dict__)

    def train(self):
        trainer = pl.Trainer(
            logger=self.logger,
            callbacks=self.callbacks if hasattr(self, 'callbacks') else None,
            # max_steps=self.args.max_steps, # 5000
            max_epochs=self.args.max_epochs, # 1000
            accelerator=self.args.accelerator, 
            log_every_n_steps=self.args.log_interval,
        )
        self.trainer = trainer
        

        self.moirai.train(trainer, self.train_set, self.val_set, self.params)

        # # save model
        # print(f"Saving model to '{self.args.save_dir}'")
        # torch.save(self.moirai.model.state_dict(), f"{self.args.save_dir}/moirai_{self.args.dataset_name}_{self.args.horizon}.pt")

    def test(self):
        labels, forecasts = self.moirai.predict(self.test_set)
        
        mse = np.mean((labels - forecasts) ** 2)
        mae = np.mean(np.abs(labels - forecasts))

        print(f"MOIRAI - MSE: {mse:.4f}, MAE: {mae:.4f}")

        # logging
        self.logger.log_metrics({'MSE': mse, 'MAE': mae})
