from .base import Base
import torch.nn as nn
from utils.utils_torch import etofts, Device
import torch

class SNN(Base):
    def __init__(self, network, lr, loggertrue, mode, weights=None, **kwargs):
        super().__init__(network=network, lr=lr, loggertrue=loggertrue, mode=mode, weights=weights)
        self.setup = "snn"
        self.loss_fn = nn.MSELoss()
        if weights != None:
            self.load_state_dict(torch.load(weights, weights_only=True, map_location=Device)['state_dict'])
        self.encoder.to(Device)

    def forward(self, x):
        params = self.encoder(x)
        x_hat = etofts(params)
        out = {"pred": params, "reconstruction": x_hat}
        return out
    
    def _get_loss(self, batch):
        x,y = batch['input'], batch['target']
        out = self.forward(x)
        x_hat = out["reconstruction"]
        params = out["pred"]
        loss = self.loss_fn(params, y)
        return loss, x_hat, params
    
    def training_step(self, batch, batch_idx):
        x,y = batch['input'], batch['target']
        loss, x_hat, params = self._get_loss(batch)
        self.log('train_loss', loss)
        self.train_metrics(x_hat, x)
        self.train_metrics_params(params, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch['input'], batch['target']
        loss, x_hat, params = self._get_loss(batch)
        self.val_metrics(x_hat, x)
        self.val_metrics_params(params, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x,y = batch['input'], batch['target']
        # loss, x_hat, params = self._get_loss(batch)
        out = self.forward(x)
        x_hat = out["reconstruction"]
        params = out["pred"]
        loss = 0
        self.test_metrics(x_hat, x)
        if not self.mode == "vivo":
            self.test_metrics_params(params, y)
            loss = self.loss_fn(params, y)
        return loss


        
   