# RNN, GRU, TCN, LSTM, DeepAR, DilatedRNN, BiTCN, MLP, NBEATS, NBEATSx, NHITS,
# DLinear, NLinear, TiDE, DeepNPTS, TFT, VanillaTransformer, Informer, Autoformer,
# FEDformer, PatchTST, TimesNet, iTransformer, StemGNN, HINT, TSMixer, TSMixerx, MLPMultivariate

from neuralforecast import models, NeuralForecast

from model.benchmarks.nf_fit import wrapper, predict_base_windows, predict_base_multivariate
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.common._base_multivariate import BaseMultivariate

import numpy as np
import inspect


def load_model(model_name):
        available_models = {name.lower(): name for name in models.__all__}
        
        # import the file containing the model class
        model_class = available_models.get(model_name.lower())

        if model_class:
            return getattr(models, model_class)

        raise ValueError(f"Model '{model_name}' not supported")


class NFHandler:
    def __init__(self, models, params, config, freq, n_series,
                 loss, optimizer, optimizer_kwargs, callbacks):
        
        NeuralForecast.fit = wrapper(args=config)
        BaseWindows.predict = predict_base_windows
        BaseMultivariate.predict = predict_base_multivariate

        self.model_names = models
        self.params = params
        self.config = config
        self.freq = freq
        self.n_series = n_series

        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.callbacks = callbacks

        self.models = []
        self.construct_models()

        self.nf = NeuralForecast(models=self.models, freq=self.freq)

        self.predict = self.nf.predict
        self.fit = self.nf.fit

    def construct_models(self):
        for i, model_name in enumerate(self.model_names):
            params = self.params[i]
            callbacks = self.callbacks[i]
            self.models.append(self._construct_model(model_name, params, callbacks))

    def _construct_model(self, model_name, params, callbacks):
        args = self.config
        model_class = load_model(model_name)

        if params is None:
            params = {'h': args.horizon, 'input_size': args.lookback}
            if 'n_series' in inspect.signature(model_class.__init__).parameters:
                params['n_series'] = self.n_series

        print(f"Creating model '{model_name}' with parameters: {params}")
        model = model_class(**params,
                            
                            # BaseModel kwargs
                            val_check_steps=args.val_interval,
                            early_stop_patience_steps=args.patience,
                            random_seed=args.seed,
                            loss=self.loss,
                            optimizer=self.optimizer,
                            optimizer_kwargs=self.optimizer_kwargs,

                            batch_size=args.batch_size,
                            max_steps=args.max_steps,
                                
                            # trainer kwargs
                            accelerator=args.accelerator,
                            devices=args.devices,
                            callbacks=callbacks,
                            log_every_n_steps=args.log_interval,
                            enable_checkpointing=True)
        
        return model
    
    def train(self, df, val_size):
        self.nf.fit(df=df, val_size=val_size)
    
    def test_predict(self, df, lookback, horizon):
        # split to lookback and forecast windows
        dates_split = df['ds'].unique()[list(range(lookback, len(df['ds'].unique()), horizon))]
        forecasts = {model_name: [] for model_name in self.model_names}
        gts = []

        for i in range(len(dates_split) - 1):
            lookback_df = df[df['ds'] < dates_split[i]]
            forecast_df = df[(df['ds'] >= dates_split[i]) & (df['ds'] < dates_split[i + 1])]

            forecast = self.nf.predict(df=lookback_df)
            for model_name in self.model_names:
                forecasts[model_name].append(forecast[model_name].values)
            gts.append(forecast_df['y'].values)

        return forecasts, np.array(gts)
    