# RNN, GRU, TCN, LSTM, DeepAR, DilatedRNN, BiTCN, MLP, NBEATS, NBEATSx, NHITS,
# DLinear, NLinear, TiDE, DeepNPTS, TFT, VanillaTransformer, Informer, Autoformer,
# FEDformer, PatchTST, TimesNet, iTransformer, StemGNN, HINT, TSMixer, TSMixerx, MLPMultivariate


def load_model(model_name):
    # import the file containing the model class
    try:
        model_module = __import__(f"neuralforecast.models.{model_name.lower()}", fromlist=[""])
    except ImportError:
        raise ImportError(f"Model '{model_name}' not found")
    
    # get the model class from the module
    model_class = getattr(model_module, model_name)

    return model_class
