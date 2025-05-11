from .base_uct import UCT
from pytorch_lightning import LightningModule
import torch.nn as nn
from utils.utils_train import Metrics, MetricsParams, MetricsParamsUncertainty
import torch
from utils.utils_torch import etofts, Device
from tqdm import tqdm
from scipy.stats import pearsonr
from utils.utils_vis import *#plot_predictions_regression, plot_uncertainty_binned, plot_paramdist_uct, plot_uncertainty_noise

class EnsembleWrapper(UCT):
    def __init__(self, ensemble_members, **kwargs):
        super().__init__(mtype="ensemble", **kwargs)#n#etwork=None, lr=None, loggertrue=None, mode=None, weights=None)
        self.ensemble_members = ensemble_members
        self.n_models = len(ensemble_members)
        self.mode = ensemble_members[0].mode
        self.loss_fn = nn.MSELoss()
        self.setup = "ensemble"
        self.setup_task()

    def _prepare_metrics_dict(self, metrics):
        for k,v in metrics.items():
            if len(v.squeeze().shape)>0:
                metrics[k] = v.mean()
        return metrics
    
    def setup_task(self):
        self.test_metrics = Metrics(num_outputs=80, prefix='test/')
        if self.mode == "sim":
            self.test_metrics_params = MetricsParams(num_outputs=4, prefix='test/')
            self.test_metrics_uct = MetricsParamsUncertainty(num_outputs=4, prefix='test/')

    def forward(self, x):
        out_params = []
        out_var = []
        for model in self.ensemble_members:
            out = model.predict_step(x)
            params = out["pred"]
            var = out["var"]
            out_params.append(params)
            out_var.append(var)
            var[var < 0] = 1e-12
        out_params = torch.stack(out_params)
        out_var = torch.stack(out_var)
        params = out_params.mean(0)
        x_hat = etofts(params)
        var = torch.mean((out_params**2+out_var), dim=0) - params **2
        var_ep = torch.mean(out_params**2, dim=0) - params**2
        var_al = torch.mean(out_var, dim=0)
        var[var < 0] = 1e-12
        assert var.isnan().sum() == 0
        out = {"pred": params, "reconstruction": x_hat, "var": var, "var_al": var_al, "var_ep": var_ep}
        return out
    
    def predict_step(self, X):
        with torch.no_grad():
            out = self.forward(X)
        return out
    
    def predict_batched(self, X, bs=512):
        dummy = self.predict_step(X[0:2])
        predictions = {k:[] for k in dummy.keys()}
        for batch in tqdm(torch.split(X, bs)):
            out = self.predict_step(batch)
            for k,v in out.items():
                predictions[k].append(v)
        for k,v in predictions.items():
            predictions[k] = torch.cat(v)
        return predictions
    
    def on_test_epoch_start(self):
        self.errors = []
        self.uncertainties = []
        self.noise_levels = []
    
    def test_step(self, batch, batch_idx):
        x = batch['input']
        y = batch['target']
        out = self.forward(x)
        x_hat = out["reconstruction"]
        params = out["pred"]
        var = out["var"]
        if self.mode == "sim":
            loss = self.loss_fn(params, y)
        else:
            loss = self.loss_fn(x_hat, x)
        self.test_metrics(x_hat, x)
        if self.mode == "sim":
            self.test_metrics_params(params, y)
            self.test_metrics_uct(params, var, y)
            error = torch.nn.functional.mse_loss(params, y, reduce=False)
            self.errors.append(error)
            self.uncertainties.append(var)
            self.noise_levels.append(batch['snr'])
        return loss
    
    def on_test_epoch_end(self):
        metrics = self._prepare_metrics_dict(self.test_metrics.compute())
        self.log_dict(metrics, on_epoch=True)
        self.test_metrics.reset()
        if self.mode == "sim":
            metrics_params = self._prepare_metrics_dict(self.test_metrics_params.compute())
            self.log_dict(metrics_params, on_epoch=True)
            self.test_metrics_params.reset()
            
            metrics_uct = self._prepare_metrics_dict(self.test_metrics_uct.compute())
            self.log_dict(metrics_uct, on_epoch=True)
            self.test_metrics_uct.reset()

            self.errors = torch.vstack(self.errors).cpu()
            self.uncertainties = torch.vstack(self.uncertainties).cpu()
            self.noise_levels = torch.cat(self.noise_levels).cpu()

            for i,p in enumerate(["ke", "dt", "ve", "vp"]):
                error_correlation = pearsonr(self.uncertainties[:,i], self.errors[:,i])[0]
                noise_correlation = pearsonr(self.uncertainties[:,i], self.noise_levels)[0]
                self.log(f"test/{p}_uct_error_correlation", error_correlation)
                self.log(f"test/{p}_uct_noise_correlation", noise_correlation)
        
    def vis_slice(self, slice_idx=1):
        return super().vis_slice(slice_idx)
        




