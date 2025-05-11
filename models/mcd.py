from .base_uct import UCT
from .snn import SNN
from .pinn import PINN
from pytorch_lightning import LightningModule
import torch.nn as nn
from utils.utils_torch import Device, etofts
import torch
from utils.utils_train import Metrics, MetricsParams, MetricsParamsUncertainty
import torch.nn.functional as F
from scipy.stats import pearsonr
from utils.utils_vis import *
from tqdm import tqdm

class MCDropoutWrapper(LightningModule):
    def __init__(self, base_model, num_samples=100):
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples
        self.setup_task(num_outputs=80)

    def setup_task(self, num_outputs):
        self.test_metrics = Metrics(num_outputs=num_outputs, prefix='test/')
        self.test_metrics_params = MetricsParams(num_outputs=4, prefix='test/')
        self.test_metrics_uct = MetricsParamsUncertainty(num_outputs=4, prefix='test/')

    def forward(self, x):
        return self.base_model.forward(x)
    
    def enable_dropout(self):
        """ Function to enable the dropout layers during inference """
        for m in self.base_model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def mc_dropout_predict(self, X):
        """ Monte Carlo Dropout inference """
        self.enable_dropout()
        predictions = {"pred": [], "reconstruction": [], "var": []}
        mc_predictions = {}
        for _ in range(self.num_samples):
            out = self.base_model.forward(X)
            predictions["pred"].append(out["pred"])
            predictions["reconstruction"].append(out["reconstruction"])
            predictions["var"].append(out["var"])
        # Aggregate predictions (e.g., by taking mean and variance)
        # predictions["var"] = (torch.stack(predictions["pred"]).std(0))**2
        mc_predictions["pred"] = torch.stack(predictions["pred"]).mean(0)
        mc_predictions["var"] = torch.mean((torch.stack(predictions["pred"])**2 + torch.stack(predictions["var"])), dim=0) - mc_predictions["pred"]**2
        mc_predictions["reconstruction"] = etofts(mc_predictions["pred"])

        return mc_predictions

    def predict_step(self, X, batch_idx=0, dataloader_idx=None):
        return self.mc_dropout_predict(X)

    def predict_batched(self, X, batchsize):
        """
        Monte Carlo Dropout batched prediction method.
        """
        self.base_model.encoder.eval()
        predictions = {"pred": [], "reconstruction": [], "var": []}
        for i in tqdm(range(0, X.shape[0], batchsize)):
            batch_preds = self.mc_dropout_predict(X[i:i+batchsize])
            predictions["pred"].append(batch_preds["pred"])
            predictions["reconstruction"].append(batch_preds["reconstruction"])
            predictions["var"].append(batch_preds["var"]) 
        
        predictions["pred"] = torch.cat(predictions["pred"], dim=0)
        predictions["reconstruction"] = torch.cat(predictions["reconstruction"], dim=0)
        predictions["var"] = torch.cat(predictions["var"], dim=0)

        return predictions

    def training_step(self, batch, batch_idx):
        return self.base_model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.base_model.validation_step(batch, batch_idx)
    
    def on_test_epoch_start(self):
        self.errors = []
        self.uncertainties = []
        self.noise_levels = []

    def test_step(self, batch, batch_idx):
        x,y,nl = batch['input'], batch['target'], batch['snr']
        predictions = self.mc_dropout_predict(x)
        x_hat, params, var = predictions["reconstruction"], predictions["pred"], predictions["var"]
        self.test_metrics(x_hat, x)
        loss = F.mse_loss(x_hat, x)
        if not self.base_model.mode == "vivo":
            self.test_metrics_params(params, y)
            self.test_metrics_uct(params, var, y)
            error = F.mse_loss(params, y, reduction='none')
            self.errors.append(error)
            self.uncertainties.append(var)
            self.noise_levels.append(nl)

        return loss
    
    def _prepare_metrics_dict(self, metrics):
        # average over all outputs
        for k,v in metrics.items():
            if len(v.squeeze().shape) > 0:
                metrics[k] = v.mean()
        return metrics
    
    def on_test_epoch_end(self):
        self.log_dict(self._prepare_metrics_dict(self.test_metrics.compute()), on_epoch=True)
        self.test_metrics.reset()
        self.log_dict(self._prepare_metrics_dict(self.test_metrics_params.compute()), on_epoch=True)
        self.test_metrics_params.reset()
        self.log_dict(self._prepare_metrics_dict(self.test_metrics_uct.compute()), on_epoch=True)
        self.test_metrics_uct.reset()

        if self.base_model.mode == "sim":
            self.errors = torch.vstack(self.errors).cpu()
            self.uncertainties = torch.vstack(self.uncertainties).cpu()
            # add a dimension to uncertainties with all zeros
            self.noise_levels = torch.cat(self.noise_levels).cpu()

            for i,p in enumerate(["ke", "dt", "ve", "vp"]):
                error_correlation = pearsonr(self.uncertainties[:,i], self.errors[:,i])[0]
                noise_correlation = pearsonr(self.uncertainties[:,i], self.noise_levels)[0]
                self.log(f"test/{p}_uct_error_correlation", error_correlation)
                self.log(f"test/{p}_uct_noise_correlation", noise_correlation)

        
    def visualise(self, datamodule, split, bs=128):
        self.to(Device)
        dataset = {
            "train": datamodule.train_dataset,
            "val": datamodule.val_dataset,
            "test": datamodule.test_dataset
        }[split]
        
        x = dataset.ct.to(Device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            predictions = self.predict_batched(x, bs)
        
        figs = {
            "accuracy": plot_predictions_regression(predictions, dataset, uct=True),
            "uct_err": plot_uncertainty_binned(predictions, dataset),
            "uct_err_smooth": plot_uncertainty_error(predictions, dataset),
            "hist": plot_paramdist_uct(predictions, dataset, datamodule),
            "uct_nl": plot_uncertainty_noise(predictions, dataset),
            "err_nl": plot_error_noise(predictions, dataset)
        }
        return figs