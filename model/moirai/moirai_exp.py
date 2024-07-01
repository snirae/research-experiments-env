import yaml
import numpy as np
import wandb
import torch
import matplotlib.pyplot as plt

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from utils.experiment import Experiment
from model.moirai.moirai import MoiraiHandler
from utils.moirai_dataset import load_dataset_for_moirai


class MoiraiExp(Experiment):
    def __init__(self, args, i):
        super(MoiraiExp, self).__init__(args)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # model parameters
        print(f"Loading model parameters from '{args.configs[i]}'")
        if args.configs[i] is not None:
            with open(args.configs[i], "r") as file:
                params = yaml.safe_load(file)
        else:
            params = None

        self.params = params

        # model
        print(f"Loading model: moirai")
        self.moirai = MoiraiHandler(
            args,
            params
        )

        model_name = f'moirai_{params["size"]}'
        if params.get('lora', False):
            model_name += '_lora'
        elif args.train:
            model_name += '_finetune'

        # callbacks
        print(f"Creating callbacks with early stopping: {args.early_stopping}, patience: {args.patience}, min improvement: {args.min_improvement}")
        callbacks = []
        
        if args.early_stopping:
            es = EarlyStopping(
                monitor='val_loss',
                patience=args.patience,
                min_delta=args.min_improvement,
                mode='min',
                verbose=True
            )
            callbacks.append(es)

        mc = ModelCheckpoint(
            dirpath=f'{args.save_dir}/{args.dataset_name}_{args.horizon}/moirai',
            monitor='val_loss',
            filename=model_name + '-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            save_weights_only=True,
        )
        callbacks.append(mc)

        self.callbacks = callbacks

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
                                 name=model_name)
        elif args.logger == "tensorboard":
            logger = TensorBoardLogger(args.log_dir,
                                       name=f'{args.dataset_name}_{args.horizon}',
                                       version=model_name)
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
            precision=32,
        )
        self.trainer = trainer
        

        self.moirai.train(trainer, self.train_set, self.val_set)

    def test(self):
        if self.args.train:
            # load best model
            print(f"Loading best model from '{self.args.save_dir}'")
            best_path = self.callbacks[1].best_model_path
            self.moirai.load_from_checkpoint(best_path)

        labels, forecasts = self.moirai.predict(self.test_set)
        
        mse = np.mean((labels - forecasts) ** 2)
        mae = np.mean(np.abs(labels - forecasts))

        print(f"MOIRAI - MSE: {mse:.4f}, MAE: {mae:.4f}")

        # logging
        self.logger.log_metrics({'MSE': mse, 'MAE': mae})

        # plot
        if self.args.save_plots:
            labels = labels.reshape(-1, labels.shape[-1])  # predction_length, num_series
            forecasts = forecasts.reshape(-1, forecasts.shape[-1])  # predction_length, num_series

            # take last 1k steps and first 15 series
            labels = labels[-1000:, :15]
            forecasts = forecasts[-1000:, :15]

            fig, ax = plt.subplots(nrows=labels.shape[-1], ncols=1, figsize=(100, 5 * labels.shape[-1]))
            for i in range(labels.shape[-1]):
                ax[i].plot(labels[:, i], label='True')
                ax[i].plot(forecasts[:, i], label='Forecast')
                ax[i].legend()
                ax[i].set_title(f'Forecast Plot - Series {i}')
                ax[i].set_xlabel('Time')
            
            plt.tight_layout()
            
            # save to logger
            if self.args.logger == 'wandb':
                self.logger.log_image(key='forecast_plot', images=[fig])
            else:
                self.logger.experiment.add_figure('forecast_plot', fig)

            print(f"Forecast plot saved.")

            plt.close()
