# 'BasePointLoss', 'MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE', 'MASE', 'relMSE', 'QuantileLoss', 'MQLoss',
# 'DistributionLoss', 'PMM', 'GMM', 'NBMM', 'HuberLoss', 'TukeyLoss', 'HuberQLoss', 'HuberMQLoss',
# 'Accuracy', 'sCRPS'

from neuralforecast.losses import pytorch
import torch
from torch import optim
import inspect


losses = {name.lower(): name for name in pytorch.__all__}

all_attrs = dir(optim)
optimizers = [attr for attr in all_attrs if inspect.isclass(getattr(optim, attr))]
optimizers = [opt for opt in optimizers if issubclass(getattr(optim, opt), optim.Optimizer)]
optimizers = {name.lower(): name for name in optimizers}


def load_loss(loss_name):
    loss = losses.get(loss_name.lower(), None)
    if loss:
        return getattr(pytorch, loss)

    raise ValueError(f"Loss '{loss_name}' not supported")
    

def load_optimizer(optimizer_name):
    optimizer = optimizers.get(optimizer_name.lower(), None)
    if optimizer:
        return getattr(optim, optimizer)

    raise ValueError(f"Optimizer '{optimizer_name}' not supported")


class LossSum(pytorch.BasePointLoss):
    def __init__(self, losses, horizon_weight=None):
        super().__init__(
            horizon_weight=horizon_weight, outputsize_multiplier=1, output_names=[""]
        )
        self.losses = losses

    def forward(self, y_hat, y, mask=None):
        return torch.sum(torch.tensor([loss(y_hat, y, mask) for loss in self.losses],
                                      requires_grad=True)
                         )/ len(self.losses)
    