from uni2ts.model.moirai import MoiraiFinetune, MoiraiForecast, MoiraiModule
from uni2ts.data.loader import DataLoader, Collate, PackCollate
from uni2ts.distribution import MixtureOutput, StudentTOutput, NormalFixedScaleOutput, NegativeBinomialOutput, LogNormalOutput

import numpy as np
import torch
from typing import Callable, Optional
from omegaconf import DictConfig
from torch.utils.data import Dataset, DistributedSampler
import lightning as pl
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation.backtest import make_evaluation_predictions


from utils.finetuning import add_lora


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset | list[Dataset]],
    ):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset
        
        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
        collate_fn: Optional[Collate] = None,
    ) -> DataLoader:
        
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            )
            if world_size > 1
            else None
        )

        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            dataloader_func=DataLoader,
            shuffle=self.cfg['train_dataloader']['shuffle'],
            world_size=self.trainer.world_size,
            batch_size=self.train_batch_size,
            # num_batches_per_epoch=self.train_num_batches_per_epoch,
            collate_fn=self.cfg['train_dataloader']['collate_fn'],
        )


    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        return self.get_dataloader(
            self.val_dataset,
            dataloader_func=DataLoader,
            shuffle=self.cfg['val_dataloader']['shuffle'],
            world_size=self.trainer.world_size,
            batch_size=self.val_batch_size,
            num_batches_per_epoch=None,
            collate_fn=self.cfg['val_dataloader']['collate_fn'],
        )
        
    @property
    def train_batch_size(self) -> int:
        return self.cfg['train_dataloader']['batch_size'] // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def val_batch_size(self) -> int:
        return self.cfg['val_dataloader']['batch_size'] // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )
    
    @property
    def train_num_batches_per_epoch(self) -> int:
        return (
            self.cfg['train_dataloader']['num_batches_per_epoch']
            * self.trainer.accumulate_grad_batches
        )


def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
    distr = self(
        **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
    )
    val_loss = self.hparams.loss_func(
        pred=distr,
        **{
            field: batch[field]
            for field in [
                "target",
                "prediction_mask",
                "observed_mask",
                "sample_id",
                "variate_id",
            ]
        },
    )
    batch_size = (
        batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
    )

    self.log(
        "val_loss",
        val_loss,
        on_step=self.hparams.log_on_step,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        sync_dist=True,
        batch_size=batch_size,
        rank_zero_only=True,
    )

    return val_loss


class MoiraiHandler:
    def __init__(self, args, size="base",
                 patch_size='auto', num_samples=100, target_dim=2,
                 lora=True):
        self.model = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{size}")
        self.args = args
        self.size = size
        self.horizon = args.horizon
        self.lookback = args.lookback
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.target_dim = target_dim

        if lora:
            self.model = add_lora(model=self.model)

        MoiraiFinetune.validation_step = validation_step
        self.finetune = MoiraiFinetune(
            min_patches=2,
            min_mask_ratio=0.15,
            max_mask_ratio=0.5,
            max_dim=128,
            module=self.model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_training_steps=args.max_steps,
            num_warmup_steps=0
        )
        self.finetune.module = self.model

        self.train_transform_map = self.finetune.train_transform_map

    def train(self, trainer, train_set, val_set, cfg):
        max_len = 512
        seq_fields = self.finetune.seq_fields
        pad_func_map = self.finetune.pad_func_map

        cfg['train_dataloader']['collate_fn'] = PackCollate(max_len, seq_fields, pad_func_map)
        cfg['train_dataloader']['batch_size'] = self.args.batch_size

        cfg['val_dataloader']['collate_fn'] = PackCollate(max_len, seq_fields, pad_func_map)
        cfg['val_dataloader']['batch_size'] = self.args.batch_size

        data_module = DataModule(cfg, train_set, val_set)
        trainer.fit(self.finetune,
                    datamodule=data_module)
    
    def predict(self, test_df, BSZ=1):
        ds = PandasDataset(test_df,
                           target=test_df.columns,
                           freq=pd.infer_freq(test_df.index))
        
        moirai_forecast = MoiraiForecast(
            module=self.model,
            prediction_length=self.horizon,
            context_length=self.lookback,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
        predictor = moirai_forecast.create_predictor(BSZ)

        forecasts = []
        tss = []
        PDT = self.horizon
        for i in range(self.lookback, len(test_df), PDT):
            context_label = test_df.iloc[i - self.lookback:i + PDT]
            ds = PandasDataset(context_label,
                               target=list(test_df.columns),
                               freq=pd.infer_freq(test_df.index))
            
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=ds,  # test dataset
                predictor=predictor,  # trained model
                num_samples=self.num_samples  # number of sample paths to draw
            ) 
            forecasts.extend(list(forecast_it))
            tss.extend(list(ts_it))

        # forecasts shape: (num_samples, num_series, prediction_length)
        # tss shape: (num_series, prediction_length)
        forecasts = np.stack([f.samples for f in forecasts]).squeeze().mean(axis=1)
        tss = np.stack([ts.values[-PDT:] for ts in tss]).squeeze()

        return tss, forecasts
    
    def load_from_checkpoint(self, checkpoint_path):
        d_model = 384 if self.size == 'small' else 1024 if self.size == 'large' else 768
        num_layers = 6 if self.size == 'small' else 24 if self.size == 'large' else 12
        self.finetune = MoiraiFinetune.load_from_checkpoint(checkpoint_path,
                                                            module_kwargs={
                                                                'distr_output': MixtureOutput(
                                                                    components=[StudentTOutput(),
                                                                                NormalFixedScaleOutput(),
                                                                                NegativeBinomialOutput(),
                                                                                LogNormalOutput()]),
                                                                'patch_sizes': tuple([8, 16, 32, 64, 128]),
                                                                'd_model': d_model,
                                                                'num_layers': num_layers,
                                                                'max_seq_len': 512,
                                                                'attn_dropout_p': 0.0,
                                                                'dropout_p': 0.0,
                                                            },
                                                            min_patches=2,
                                                            min_mask_ratio=0.15,
                                                            max_mask_ratio=0.5,
                                                            max_dim=128,
                                                            lr=self.args.lr,
                                                            weight_decay=self.args.weight_decay,
                                                            num_training_steps=self.args.max_steps,
                                                            num_warmup_steps=0,
                                                            strict=False)
        self.finetune.module = self.model
        
        print(f"Loaded best model")
