from .base_uct import UCT
from .base import Base
from networks import get_network
from .pinn import PINN
import torch.nn as nn
from utils.utils_torch import etofts, Device
import torch
from torch.func import jacrev
from functools import partial
from torch import vmap
from utils.utils_train import MetricsParamsUncertainty
from utils.utils_vis import *

# import torch.autograd.functional as AF
class PINN_PH(Base):
    def __init__(self, network, lr, loggertrue, mode, weights=None, **kwargs):
        super().__init__(network=network, lr=lr, loggertrue=loggertrue, mode=mode, weights=None, **kwargs)
        self.setup = "pinn_ph"
        self.burnin_epochs = kwargs.get("burnin_epochs", 0)
        self.encoder.to(Device)
        self.inference = kwargs.get("inference", False)
        self.mode = mode
        self.uct_model = UCT(network, lr, loggertrue, mode, weights=None, **kwargs)
        self.predictions = {}
        if weights != None:
            self.load_state_dict(torch.load(weights, weights_only=True, map_location=Device)['state_dict'])

    def set_inference(self, inference):
        self.inference = inference
        if inference == True:
            # freeze encoder
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def forward(self, x):
        params = self.encoder(x)
        x_hat = etofts(params)

        if not self.inference:
            out = {"pred": params, "reconstruction": x_hat}
            return out
        
        if self.inference:
            res = x - x_hat
            cov = self.inv_cov(params, res)
            var = cov.diagonal(dim1=-2, dim2=-1)
            # clip var
            var = torch.clamp(var, min=0)
            out = {"pred": params, "reconstruction": x_hat, "cov": cov, "var": var}
            return out
        
    def _get_loss(self, batch):
        x = batch['input']
        out = self.forward(x)
        x_hat = out["reconstruction"]
        if not out["pred"].requires_grad:
            out["pred"].requires_grad = True
        res = x - x_hat
        loss = (res**2).mean()
        return loss, x_hat, out

    def inv_cov(self, params, res):
        assert res.shape == (params.shape[0], 80)
        sigma = torch.sum(res**2, dim=-1) / (77) # N 
        # ensure gradients are on
        with torch.enable_grad():
            params2 = params.clone().to(Device).requires_grad_()
            assert params2.requires_grad
            jacobian = vmap(jacrev(self.tofts_wrapper), in_dims=(0))(params2).squeeze() # N x 80 x 4
            # if not self.testing: #NOTE: During training, the jacobian calculatinon
            assert jacobian.sum() > 0
            # delete gradients
            jacobian = jacobian.detach()

        hessian_approx = torch.bmm(jacobian.transpose(1,2), jacobian) # N x 4 x 4
        # reg = "pinv"
        cov_params = sigma.unsqueeze(1).unsqueeze(1) * torch.pinverse(hessian_approx)
        return cov_params
    
    def tofts_wrapper(self, x):
        out = etofts(x)
        return out
    
    def training_step(self, batch, batch_idx):
        loss, x_hat, out = self._get_loss(batch)
        self.train_metrics(x_hat, batch['input'])
        if self.mode == "sim":
            self.train_metrics_params(out["pred"], batch['target'])
        self.log('train_loss', loss.mean())
        return loss.mean()
    
    def validation_step(self, batch, batch_idx):
        loss, x_hat, out = self._get_loss(batch)
        self.val_metrics(x_hat, batch['input'])
        if self.mode =="sim":
            self.val_metrics_params(out["pred"], batch['target'])
            self.val_metrics_uct(out["pred"], out["var"], batch['target'])
        self.log('val_loss', loss.mean())
        return loss.mean()
    
    def test_step(self, batch, batch_idx):
        self.eval()
        with torch.enable_grad():
            loss, x_hat, out = self._get_loss(batch)
            self.log('test_loss', loss.mean())
        self.test_metrics(x_hat, batch['input'])
        if self.mode == "sim":
            self.test_metrics_params(out["pred"], batch['target'])
            # self.test_metrics_uct(out["pred"], out["var"], batch['target'])
        return loss.mean()
    
    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.set_inference(False)

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        self.set_inference(False)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.set_inference(True)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self.set_inference(False)

    def setup_task(self, num_outputs):
        super().setup_task(num_outputs)
        if self.mode == "sim":
            # self.train_metrics_uct = MetricsParamsUncertainty(num_outputs=4, prefix="train/")
            self.val_metrics_uct = MetricsParamsUncertainty(num_outputs=4, prefix="val/")
            # self.test_metrics_uct = MetricsParamsUncertainty(num_outputs=4, prefix="test/")

    def predict_batched(self, X, bs):
        self.set_inference(True)
        # super predict batched
        predictions = super().predict_batched(X, bs)
        return predictions
    
    def predict_step(self, x):
        self.set_inference(True)
        predictions = super().predict_step(x)
        return predictions
    
    def visualise(self, datamodule, split='test', bs=512):
        self.set_inference(True)
        dataset = {
            "train": datamodule.train_dataset,
            "val": datamodule.val_dataset,
            "test": datamodule.test_dataset
        }[split]
        if self.mtype != "ensemble":
            self.encoder.to(Device)
        figs = super().visualise(datamodule=datamodule, split=split, bs=bs)
        if self.mode == "sim":
            uct_err = plot_uncertainty_binned(self.predictions[split], dataset)
            figs["uct_err"] = uct_err
            figs["uct_err_smooth"] = plot_uncertainty_error(self.predictions[split], dataset)
            figs["hist"] = plot_paramdist_uct(self.predictions[split], dataset, datamodule)
            figs["uct_nl"] = plot_uncertainty_noise(self.predictions[split], dataset)

        # figs["err_nl"] = plot_error_noise(self.predictions[split], dataset)
        return figs
        return figs
        



        
