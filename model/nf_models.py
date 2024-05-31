# RNN, GRU, TCN, LSTM, DeepAR, DilatedRNN, BiTCN, MLP, NBEATS, NBEATSx, NHITS,
# DLinear, NLinear, TiDE, DeepNPTS, TFT, VanillaTransformer, Informer, Autoformer,
# FEDformer, PatchTST, TimesNet, iTransformer, StemGNN, HINT, TSMixer, TSMixerx, MLPMultivariate

from neuralforecast import models


models = {name.lower(): name for name in models.__all__}

def load_model(model_name):
    # import the file containing the model class
    model_class = models.get(model_name.lower())

    if model_class:
        return getattr(models, model_class)

    raise ValueError(f"Model '{model_name}' not supported")
