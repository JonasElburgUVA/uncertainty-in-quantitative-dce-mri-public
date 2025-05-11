from .base import Base
import torch.nn as nn
from utils.utils_torch import etofts, Device
import torch

class PINN(Base):
    def __init__(self, network, lr, loggertrue, mode, weights=None, **kwargs):
        super().__init__(network=network, lr=lr, loggertrue=loggertrue, mode=mode, weights=weights)
        self.setup = "pinn"
        self.loss_fn = nn.MSELoss()
        if weights != None:
            self.load_state_dict(torch.load(weights, map_location=Device, weights_only=True)['state_dict'])
        self.encoder.to(Device)
 
    def forward(self, x):
        params = self.encoder(x)
        x_hat = etofts(params)
        out = {"pred": params, "reconstruction": x_hat}
        return out
    
    def _get_loss(self, batch):
        x = batch['input']
        out = self.forward(x)
        x_hat = out["reconstruction"]
        params = out["pred"]
        loss = self.loss_fn(x_hat, x)
        return loss, x_hat, params
    
    def training_step(self, batch, batch_idx):
        x = batch['input']
        loss, x_hat, params = self._get_loss(batch)
        self.log('train_loss', loss)
        self.train_metrics(x_hat, x)
        if self.mode == "sim":
            y = batch['target']
            self.train_metrics_params(params, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['input']
        loss, x_hat, params = self._get_loss(batch)
        self.log('val_loss', loss)
        self.val_metrics(x_hat, x)
        if self.mode == "sim":
            y = batch['target']
            self.val_metrics_params(params, y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['input']
        loss, x_hat, params = self._get_loss(batch)
        self.log('test_loss', loss)
        self.test_metrics(x_hat, x)

        if self.mode == "sim":
            y = batch['target']
            self.test_metrics_params(params, y)

        return loss
    
        
    
    



