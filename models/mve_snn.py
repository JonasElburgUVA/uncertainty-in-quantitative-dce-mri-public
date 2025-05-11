from .base_uct import UCT
import torch.nn as nn
from utils.utils_torch import etofts, Device
import torch

class MVE_SNN(UCT):
    def __init__(self, network, lr, loggertrue, mode, weights=None, **kwargs):
        super().__init__(network=network, lr=lr, loggertrue=loggertrue, mode=mode, weights=weights, **kwargs)
        self.setup = "mve_snn"
        self.loss_fn = nn.GaussianNLLLoss()
        self.burnin_epochs = kwargs.get("burnin_epochs", 0)

        if weights != None:
            self.load_state_dict(torch.load(weights, map_location=Device, weights_only=True)['state_dict'])

        self.encoder.to(Device)

    def forward(self, x):
        out = self.encoder(x)
        params = out[:,0,:]
        log_var = out[:,1,:]
        var = torch.exp(log_var)
        x_hat = etofts(params)
        out = {"pred": params, "reconstruction": x_hat, "var": var}
        return out
    
    def _get_loss(self, batch):
        x,y = batch['input'], batch['target']
        out = self.forward(x)
        x_hat = out["reconstruction"]
        params = out["pred"]
        var = out["var"]
        if self.current_epoch < self.burnin_epochs:
            loss = torch.nn.functional.mse_loss(x_hat, x)
        else:
            loss = self.loss_fn(params, y, var)
        return loss, x_hat, params, var
    
    def training_step(self, batch, batch_idx):
        x,y = batch['input'], batch['target']
        loss, x_hat, params, var = self._get_loss(batch)
        self.log('train_loss', loss)
        self.train_metrics(x_hat, x)
        self.train_metrics_params(params, y)
        self.train_metrics_uct(params, var, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch['input'], batch['target']
        loss, x_hat, params, var = self._get_loss(batch)
        self.val_metrics(x_hat, x)
        self.val_metrics_params(params, y)
        self.val_metrics_uct(params, var, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x,y,nl = batch['input'], batch['target'], batch['snr']
        loss, x_hat, params, var = self._get_loss(batch)
        self.test_metrics(x_hat, x)
        if not self.mode == "vivo":
            self.test_metrics_params(params, y)
            self.test_metrics_uct(params, var, y)
            error = torch.nn.functional.mse_loss(params, y, reduce=False)
            self.errors.append(error)
            self.uncertainties.append(var)  
            self.noise_levels.append(nl)
        return loss