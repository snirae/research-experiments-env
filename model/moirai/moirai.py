from uni2ts.model.moirai import MoiraiFinetune, MoiraiForecast, MoiraiModule
from uni2ts.distribution import MixtureOutput, StudentTOutput, NormalFixedScaleOutput, NegativeBinomialOutput, LogNormalOutput
from uni2ts.loss.packed import PackedNLLLoss, PackedMSELoss
from uni2ts.data.loader import DataLoader, Collate

import numpy as np
from typing import Callable, Optional
from omegaconf import DictConfig
from torch.utils.data import Dataset, DistributedSampler
import pytorch_lightning as pl
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation.backtest import make_evaluation_predictions

from model.finetuning import add_lora


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


class MoiraiHandler:
    def __init__(self, args, size="base",
                 patch_size='auto', num_samples=1, target_dim=2,
                 lora=True):
        self.model = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{size}")
        self.size = size
        self.horizon = args.horizon
        self.lookback = args.lookback
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.target_dim = target_dim

        if lora:
            self.model = add_lora(model=self.model)

        self.finetune = MoiraiFinetune(
            module_kwargs={
                "distr_output": MixtureOutput(
                        [StudentTOutput(), NormalFixedScaleOutput(),
                         NegativeBinomialOutput(), LogNormalOutput()]
                ),
                
                "d_model": 384,
                "num_layers": 6,
                "patch_sizes": [8, 16, 32, 64, 128],
                "max_seq_len": 512,
                "attn_dropout_p": 0.0,
                "dropout_p": 0.0,
                "scaling": True,
            },
            loss_func=PackedNLLLoss(),
            val_metric=PackedMSELoss(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            beta1=0.9,
            beta2=0.98,
            num_training_steps=args.max_steps,
            num_warmup_steps=0
        )
        self.finetune.module = self.model

    def train(self, trainer, train_set, val_set, cfg):
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
                num_samples=1  # number of sample paths to draw
            ) 
            forecasts.extend(list(forecast_it))
            tss.extend(list(ts_it))

        forecasts = np.stack([f.samples[0] for f in forecasts])
        tss = np.stack([ts.values[-PDT:] for ts in tss])

        forecasts = forecasts.reshape(-1, forecasts.shape[-1])
        tss = tss.reshape(-1, tss.shape[-1])

        return tss, forecasts


# yaml config file of moiraifinetune
# target: uni2ts.model.moirai.MoiraiFinetune
# module:
#   target: uni2ts.model.moirai.MoiraiModule.from_pretrained
#   pretrained_model_name_or_path: Salesforce/moirai-1.0-R-small
# module_kwargs:
#   target: builtins.dict
#   distr_output:
#     target: uni2ts.distribution.MixtureOutput
#     components:
      
# target: uni2ts.distribution.StudentTOutput
# target: uni2ts.distribution.NormalFixedScaleOutput
# target: uni2ts.distribution.NegativeBinomialOutput
# target: uni2ts.distribution.LogNormalOutput
# d_model: 384
# num_layers: 6
# patch_sizes: ${as_tuple:[8, 16, 32, 64, 128]}
# max_seq_len: 512
# attn_dropout_p: 0.0
# dropout_p: 0.0
# scaling: true
# min_patches: 2
# min_mask_ratio: 0.15
# max_mask_ratio: 0.5
# max_dim: 128
# loss_func:
#   target: uni2ts.loss.packed.PackedNLLLoss
# val_metric:
  
# target: uni2ts.loss.packed.PackedMSELoss
# target: uni2ts.loss.packed.PackedNRMSELoss
# normalize: absolute_target_squared
# lr: 1e-3
# weight_decay: 1e-1
# beta1: 0.9
# beta2: 0.98
# num_training_steps: ${mul:${trainer.max_epochs},${train_dataloader.num_batches_per_epoch}}
# num_warmup_steps: 0