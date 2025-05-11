from .base_uct import UCT
from networks import get_network
from .pinn_ph import PINN_PH
import torch.nn as nn
from utils.utils_torch import etofts, Device
import torch
from torch.autograd.functional import jacobian
from torch.func import jacrev
from torch import vmap
from functools import partial

class MVE_PINN(UCT):
    def __init__(self, network, lr, loggertrue, mode, weights=None, **kwargs):
        super().__init__(network=network, lr=lr, loggertrue=loggertrue, mode=mode, weights=None, **kwargs)
        self.setup = "mve_pinn"
        self.loss_fn = nn.GaussianNLLLoss()
        self.burnin_epochs = kwargs.get("burnin_epochs", 0)

        if kwargs["dt_predictor"] is not None:
            network = get_network('dcenet')
            self.dt_predictor = PINN_PH(network=network, lr=lr, loggertrue=False, mode=mode, weights=kwargs["dt_predictor"])
            self.dt_predictor.to(Device)
            self.dt_predictor.encoder.to(Device)
        else:
            network = get_network('dcenet')
            self.dt_predictor = PINN_PH(network=network, lr=lr, loggertrue=False, mode=mode)
        self.dt_predictor.to(Device)
        self.dt_predictor.encoder.to(Device)
        self.dt_predictor.set_inference(False)

        if weights != None:
            self.load_state_dict(torch.load(weights, map_location=Device, weights_only=True)['state_dict'])
        self.encoder.to(Device)


    def forward(self, x, batch=None):
        out = self.encoder(x)
        params = out[:,0,:]
        log_var = out[:,1]
        with torch.no_grad():
            params_det = self.dt_predictor.encoder(x)
        params[:,1] = params_det[:,1].detach()
        var_z = torch.exp(log_var)
        mask = torch.ones_like(var_z)
        mask[:,1] = 0
        var_z = var_z * mask
        x_hat = etofts(params)

         
        if self.current_epoch < self.burnin_epochs:
            out = {"pred": params, "reconstruction": x_hat, "var": var_z, "params_det" : params_det}
            return out
        
        else: 
            jac = vmap(jacrev(etofts), in_dims=(0))(params).squeeze().detach() # shape (batch, params, input)
            jac_abs = jac.abs()
            # Some numerical stability steps
            jac_abs = torch.maximum(jac_abs, torch.ones_like(jac)*1e-8)#.double()
            jac_abs = torch.minimum(jac_abs, torch.ones_like(jac)*1e5)#.double()
            # zero out dt
            jac_abs[:,:,1] = torch.zeros_like(jac_abs[:,:,1])
            var_z[:,[0,2,3]] = torch.maximum(var_z[:,[0,2,3]], torch.ones_like(var_z[:,[0,2,3]])*1e-8)#.double()
            
            # approximate uncertainty propagation
            var_x = torch.bmm(jac_abs.square(), var_z.unsqueeze(2)).squeeze()
            assert var_x.isnan().sum() == 0, "NaN detected in var_x during forward pass"

            out = {"pred": params, "reconstruction": x_hat, "var": var_z, "var_x": var_x, "jacobian": jac, "params_det" : params_det}
            return out
    
    def _get_loss(self, batch):
        x = batch['input']
        out = self.forward(x)
        x_hat = out["reconstruction"]
        params = out["pred"]
        var_z = out["var"]
        params_det = out["params_det"]
        regularisation = (params - params_det).abs().exp().mean()

        if self.current_epoch < self.burnin_epochs:
            loss = torch.nn.functional.mse_loss(x_hat, x) + (var_z-torch.ones_like(var_z)*0.1).abs().mean() + 5.0 * regularisation
            return loss, x_hat, params, var_z, None, regularisation
        else:
            var_x = out["var_x"]
            mask = x_hat == 0
            num = (x_hat - x).pow(2)
            den = torch.where(mask, torch.ones_like(var_x), var_x)
            loss = ((num/den) + torch.log(den))[~mask].mean() + 5.0 * regularisation# + (mse_pre_contrast).mean
            if loss != loss or loss == float('inf') or loss == float('-inf'):
                print("Warning: NaN detected in loss")

            return loss, x_hat, params, var_z, var_x, regularisation
    
    def training_step(self, batch, batch_idx):
        loss, x_hat, params, var_z, var_x, reg = self._get_loss(batch)
        self.log('train_loss', loss)
        self.log('reg_loss', reg)
        self.train_metrics(x_hat, batch['input'])
        if not self.mode == "vivo":
            self.train_metrics_params(params, batch['target'])
            self.train_metrics_uct(params, var_z, batch['target'])
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        loss, x_hat, params, var_z, var_x, reg = self._get_loss(batch)
        self.val_metrics(x_hat, batch['input'])
        if not self.mode == "vivo":
            self.val_metrics_params(params, batch['target'])
            self.val_metrics_uct(params, var_z, batch['target'])
        self.log('val_loss', loss)
        self.log('reg_loss', reg)
        return loss
    

    def test_step(self, batch, batch_idx):
        self.eval()
        x,y = batch['input'], batch['target']
        loss, x_hat, params, var_z, var_x, reg = self._get_loss(batch)
        self.test_metrics(x_hat, x)
        if not self.mode == "vivo":
            self.test_metrics_params(params, y)
            self.test_metrics_uct(params, var_z, y)
            error = torch.nn.functional.mse_loss(params, y, reduce=False)
            self.errors.append(error)
            self.uncertainties.append(var_z)
            nl = batch['snr']
            self.noise_levels.append(nl)
        self.log('test_loss', loss)
        self.log('reg_loss', reg)
        return loss
    
