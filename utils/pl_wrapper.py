# PyTorch Lightning wrapper for training models

import pytorch_lightning as pl
import torch


class TrainWrapper(pl.LightningModule):
    def __init__(self, model, losses, optimizer, lr, weight_decay):
        super(TrainWrapper, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        self.losses = []
        for loss in losses:
            if loss.lower() == "mse":
                self.losses.append(torch.nn.MSELoss())
            elif loss.lower() == "mae":
                self.losses.append(torch.nn.L1Loss())
            else:
                raise ValueError(f"Loss function '{loss}' not supported")
        
        self.loss_fn = lambda y_hat, y: sum([loss(y_hat, y) for loss in self.losses]) / len(self.losses)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' not supported")
        
        return optimizer
    