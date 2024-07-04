from uni2ts.model.moirai import MoiraiFinetune, MoiraiForecast, MoiraiModule
from uni2ts.data.loader import DataLoader, Collate, PackCollate
from uni2ts.module.packed_scaler import PackedNOPScaler
from uni2ts.eval_util.evaluation import evaluate_model

import numpy as np
import torch
from typing import Callable, Optional
from omegaconf import DictConfig
from torch.utils.data import Dataset, DistributedSampler
import lightning as pl
import pandas as pd

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.ev.metrics import MSE, MAE
from gluonts.time_feature import get_seasonality

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
    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.horizon = args.horizon
        self.lookback = args.lookback
        
        if params is not None:
            self.size = params.get("size", "small")
            self.patch_size = params.get("patch_size", "auto")
            self.num_samples = params.get("num_samples", 100)
            self.target_dim = params.get("target_dim", 2)
            self.lora = params.get("lora", False)
        else:
            self.size = "small"
            self.patch_size = "auto"
            self.num_samples = 100
            self.target_dim = 2
            self.lora = False

        self.model = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{self.size}")
        # self.model.scaler = PackedNOPScaler()
        if self.lora:
            self.model = add_lora(model=self.model)

        MoiraiFinetune.validation_step = validation_step
        self.finetune = MoiraiFinetune(
            min_patches=1,  # 2,
            min_mask_ratio=0.15,
            max_mask_ratio=0.5,
            max_dim=1000,  # 128,
            module=self.model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_training_steps=args.max_steps,
            num_warmup_steps=0
        )
        self.finetune.module = self.model

        self.train_transform_map = self.finetune.train_transform_map
        self.val_transform_map = self.finetune.val_transform_map

    def train(self, trainer, train_set, val_set):
        if self.params is not None:
            cfg = self.params
        else:
            cfg = DictConfig(
                {
                    "train_dataloader": {
                        "batch_size": self.args.batch_size,
                        "shuffle": True,
                        "num_batches_per_epoch": None,
                        "collate_fn": None,
                    },
                    "val_dataloader": {
                        "batch_size": self.args.batch_size,
                        "shuffle": False,
                        "num_batches_per_epoch": None,
                        "collate_fn": None,
                    },
                }
            )
        
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
        predictor = moirai_forecast.create_predictor(BSZ, self.args.accelerator)

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
                num_samples=self.num_samples,  # number of sample paths to draw
            )
            forecasts.extend(list(forecast_it))
            tss.extend(list(ts_it))

        # forecasts shape: (num_samples, num_series, prediction_length)
        # tss shape: (num_series, prediction_length)
        forecasts = np.stack([f.samples for f in forecasts]).squeeze().mean(axis=1)
        tss = np.stack([ts.values[-PDT:] for ts in tss]).squeeze()

        return tss, forecasts
    
    def load_from_checkpoint(self, checkpoint_path):
        self.finetune.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        self.model = self.finetune.module

    def get_moirai_forecast(self, metadata):
        return MoiraiForecast(
            module=self.model,
            prediction_length=self.horizon,
            context_length=self.lookback,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )

    def evaluate(self, test_set, metadata, min_batch_size=1):
        batch_size = self.args.batch_size
        while True:
            model = self.get_moirai_forecast(metadata)
            metrics = [MSE(), MAE()]
            try:
                predictor = model.create_predictor(batch_size, self.args.accelerator)
                res = evaluate_model(
                    predictor,
                    test_data=test_set,
                    metrics=metrics,
                    batch_size=batch_size,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=get_seasonality(metadata.freq),
                )
                print(res)

                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}"
                )
                batch_size //= 2
                if batch_size < min_batch_size:
                    print(
                        f"batch_size {batch_size} smaller than "
                        f"min_batch_size {min_batch_size}, ending evaluation"
                    )
                    break
            
        return res
