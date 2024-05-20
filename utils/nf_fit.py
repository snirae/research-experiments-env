import warnings
from typing import Optional, Union

import fsspec
import pandas as pd
import numpy as np
from utilsforecast.compat import DataFrame, pl_DataFrame
from utilsforecast.validation import validate_freq

from neuralforecast.common._base_model import DistributedConfig
from neuralforecast.compat import SparkDataFrame
from neuralforecast.tsdataset import _FilesDataset, TimeSeriesDataModule

import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import torch
import pytorch_lightning as pl


def wrapper(args):
    
    def fit(
            self,
            df: Optional[Union[DataFrame, SparkDataFrame]] = None,
            static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
            val_size: Optional[int] = 0,
            sort_df: bool = True,
            use_init_models: bool = False,
            verbose: bool = False,
            id_col: str = "unique_id",
            time_col: str = "ds",
            target_col: str = "y",
            distributed_config: Optional[DistributedConfig] = None,
        ) -> None:
            """Fit the core.NeuralForecast.

            Fit `models` to a large set of time series from DataFrame `df`.
            and store fitted models for later inspection.

            Parameters
            ----------
            df : pandas, polars or spark DataFrame, optional (default=None)
                DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
                If None, a previously stored dataset is required.
            static_df : pandas, polars or spark DataFrame, optional (default=None)
                DataFrame with columns [`unique_id`] and static exogenous.
            val_size : int, optional (default=0)
                Size of validation set.
            sort_df : bool, optional (default=False)
                Sort `df` before fitting.
            use_init_models : bool, optional (default=False)
                Use initial model passed when NeuralForecast object was instantiated.
            verbose : bool (default=False)
                Print processing steps.
            id_col : str (default='unique_id')
                Column that identifies each serie.
            time_col : str (default='ds')
                Column that identifies each timestep, its values can be timestamps or integers.
            target_col : str (default='y')
                Column that contains the target.
            distributed_config : neuralforecast.DistributedConfig
                Configuration to use for DDP training. Currently only spark is supported.

            Returns
            -------
            self : NeuralForecast
                Returns `NeuralForecast` class with fitted `models`.
            """
            if (df is None) and not (hasattr(self, "dataset")):
                raise Exception("You must pass a DataFrame or have one stored.")

            # Model and datasets interactions protections
            if (any(model.early_stop_patience_steps > 0 for model in self.models)) and (
                val_size == 0
            ):
                raise Exception("Set val_size>0 if early stopping is enabled.")

            # Process and save new dataset (in self)
            if isinstance(df, (pd.DataFrame, pl_DataFrame)):
                validate_freq(df[time_col], self.freq)
                self.dataset, self.uids, self.last_dates, self.ds = self._prepare_fit(
                    df=df,
                    static_df=static_df,
                    sort_df=sort_df,
                    predict_only=False,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                )
                self.sort_df = sort_df
            elif isinstance(df, SparkDataFrame):
                if distributed_config is None:
                    raise ValueError(
                        "Must set `distributed_config` when using a spark dataframe"
                    )
                if self.local_scaler_type is not None:
                    raise ValueError(
                        "Historic scaling isn't supported in distributed. "
                        "Please open an issue if this would be valuable to you."
                    )
                temporal_cols = [c for c in df.columns if c not in (id_col, time_col)]
                if static_df is not None:
                    if not isinstance(static_df, SparkDataFrame):
                        raise ValueError(
                            "`static_df` must be a spark dataframe when `df` is a spark dataframe."
                        )
                    static_cols = [c for c in static_df.columns if c != id_col]
                    df = df.join(static_df, on=[id_col], how="left")
                else:
                    static_cols = None
                self.id_col = id_col
                self.time_col = time_col
                self.target_col = target_col
                self.scalers_ = {}
                self.sort_df = sort_df
                num_partitions = distributed_config.num_nodes * distributed_config.devices
                df = df.repartitionByRange(num_partitions, id_col)
                df.write.parquet(path=distributed_config.partitions_path, mode="overwrite")
                fs, _, _ = fsspec.get_fs_token_paths(distributed_config.partitions_path)
                protocol = fs.protocol
                if isinstance(protocol, tuple):
                    protocol = protocol[0]
                files = [
                    f"{protocol}://{file}"
                    for file in fs.ls(distributed_config.partitions_path)
                    if file.endswith("parquet")
                ]
                self.dataset = _FilesDataset(
                    files=files,
                    temporal_cols=temporal_cols,
                    static_cols=static_cols,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    min_size=df.groupBy(id_col).count().agg({"count": "min"}).first()[0],
                )
            elif df is None:
                if verbose:
                    print("Using stored dataset.")
            else:
                raise ValueError(
                    f"`df` must be a pandas, polars or spark DataFrame or `None`, got: {type(df)}"
                )

            if val_size is not None:
                if self.dataset.min_size < val_size:
                    warnings.warn(
                        "Validation set size is larger than the shorter time-series."
                    )

            # Recover initial model if use_init_models
            if use_init_models:
                self._reset_models()

            for i, model in enumerate(self.models):
                
                if args.logger == "wandb":
                    with open(args.api_key_file, "r") as file:
                        api_key = file.read().strip()
                    wandb.login(key=api_key)
                    
                    logger = WandbLogger(save_dir=args.log_dir,
                                        project=args.project if args.project else args.dataset_name,
                                        entity=args.entity,
                                        name=args.models[i])
                elif args.logger == "tensorboard":
                    logger = TensorBoardLogger(args.log_dir,
                                            name=args.dataset_name,
                                            version=args.models[i])
                else:
                    raise ValueError(f"Logger '{args.logger}' not supported")
                
                logger.log_hyperparams(vars(args))
                model.trainer_kwargs['logger'] = logger

                self.models[i] = model.fit(
                    self.dataset, val_size=val_size, distributed_config=distributed_config
                )

                # close logger
                if args.logger == "wandb":
                    logger.experiment.finish()
                elif args.logger == "tensorboard":
                    logger.experiment.close()

            self._fitted = True

    return fit


def predict_base_windows(
        self,
        dataset,
        test_size=None,
        step_size=1,
        random_seed=None,
        **data_module_kwargs,
    ):
        """Predict.

        Neural network prediction with PL's `Trainer` execution of `predict_step`.

        **Parameters:**<br>
        `dataset`: NeuralForecast's `TimeSeriesDataset`, see [documentation](https://nixtla.github.io/neuralforecast/tsdataset.html).<br>
        `test_size`: int=None, test size for temporal cross-validation.<br>
        `step_size`: int=1, Step size between each window.<br>
        `random_seed`: int=None, random_seed for pytorch initializer and numpy generators, overwrites model.__init__'s.<br>
        `**data_module_kwargs`: PL's TimeSeriesDataModule args, see [documentation](https://pytorch-lightning.readthedocs.io/en/1.6.1/extensions/datamodules.html#using-a-datamodule).
        """
        self._check_exog(dataset)
        self._restart_seed(random_seed)

        self.predict_step_size = step_size
        self.decompose_forecast = False
        datamodule = TimeSeriesDataModule(
            dataset=dataset,
            valid_batch_size=self.valid_batch_size,
            **data_module_kwargs,
        )

        # Protect when case of multiple gpu. PL does not support return preds with multiple gpu.
        pred_trainer_kwargs = self.trainer_kwargs.copy()
        if (pred_trainer_kwargs.get("accelerator", None) == "gpu") and (
            torch.cuda.device_count() > 1
        ):
            pred_trainer_kwargs["devices"] = [0]

        pred_trainer_kwargs.pop('logger', None)

        trainer = pl.Trainer(**pred_trainer_kwargs)
        fcsts = trainer.predict(self, datamodule=datamodule)
        fcsts = torch.vstack(fcsts).numpy().flatten()
        fcsts = fcsts.reshape(-1, len(self.loss.output_names))
        return fcsts


def predict_base_multivariate(
        self,
        dataset,
        test_size=None,
        step_size=1,
        random_seed=None,
        **data_module_kwargs,
    ):
        """Predict.

        Neural network prediction with PL's `Trainer` execution of `predict_step`.

        **Parameters:**<br>
        `dataset`: NeuralForecast's `TimeSeriesDataset`, see [documentation](https://nixtla.github.io/neuralforecast/tsdataset.html).<br>
        `test_size`: int=None, test size for temporal cross-validation.<br>
        `step_size`: int=1, Step size between each window.<br>
        `**data_module_kwargs`: PL's TimeSeriesDataModule args, see [documentation](https://pytorch-lightning.readthedocs.io/en/1.6.1/extensions/datamodules.html#using-a-datamodule).
        """
        self._check_exog(dataset)
        self._restart_seed(random_seed)

        self.predict_step_size = step_size
        self.decompose_forecast = False
        datamodule = TimeSeriesDataModule(
            dataset=dataset,
            valid_batch_size=self.n_series,
            batch_size=self.n_series,
            **data_module_kwargs,
        )

        # Protect when case of multiple gpu. PL does not support return preds with multiple gpu.
        pred_trainer_kwargs = self.trainer_kwargs.copy()
        if (pred_trainer_kwargs.get("accelerator", None) == "gpu") and (
            torch.cuda.device_count() > 1
        ):
            pred_trainer_kwargs["devices"] = [0]

        pred_trainer_kwargs.pop('logger', None)

        trainer = pl.Trainer(**pred_trainer_kwargs)
        fcsts = trainer.predict(self, datamodule=datamodule)
        fcsts = torch.vstack(fcsts).numpy()

        fcsts = np.transpose(fcsts, (2, 0, 1))
        fcsts = fcsts.flatten()
        fcsts = fcsts.reshape(-1, len(self.loss.output_names))
        return fcsts
