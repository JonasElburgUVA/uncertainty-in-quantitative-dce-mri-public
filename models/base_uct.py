from .base import Base
import torch.nn as nn
from utils.utils_torch import etofts, Device
from utils.utils_vis import *
import torch
from tqdm import tqdm
from utils.utils_train import MetricsParamsUncertainty
from scipy.stats import pearsonr
from networks import get_network


class UCT(Base):
    def __init__(self, 
                 network=get_network("fc"),
                 lr=1e-4,
                 loggertrue=False,
                 mode="sim",
                 mtype="uct",
                 weights=None, 
                 out_path=None,
                 **kwargs):
        super().__init__(network=network, lr=lr, loggertrue=loggertrue, mode=mode, weights=weights, mtype=mtype, out_path=out_path, **kwargs)
        self.burnin_epochs = None

    def on_train_epoch_end(self):
        """
        Log metrics at the end of each training epoch
        """
        super().on_train_epoch_end()
        if self.mode == "sim":
            self.log_dict(self._prepare_metrics_dict(self.train_metrics_uct.compute()), on_epoch=True)
            self.train_metrics_uct.reset()

    def on_validation_epoch_end(self):
        """
        Log metrics at the end of each validation epoch after burnin
        """
        super().on_validation_epoch_end()
        print(f"Validation epoch {self.current_epoch} ending")
        if self.mode == "sim":
            self.log_dict(self._prepare_metrics_dict(self.val_metrics_uct.compute()), on_epoch=True)
            self.val_metrics_uct.reset()
        if self.current_epoch == self.burnin_epochs and self.trainer.check_val_every_n_epoch != 1:
            self.trainer.check_val_every_n_epoch = 1
            print(f"Setting check_val_every_n_epoch to 1 at epoch {self.current_epoch}")

    def on_test_epoch_end(self):
        """
        Log metrics at the end of each test epoch
        """
        super().on_test_epoch_end()
        if self.mode == "sim":
            self.log_dict(self._prepare_metrics_dict(self.test_metrics_uct.compute()), on_epoch=True)
            self.errors = torch.vstack(self.errors).cpu()
            self.uncertainties = torch.vstack(self.uncertainties).cpu()
            self.noise_levels = torch.cat(self.noise_levels).cpu()
            for i,p in enumerate(["ke", "dt", "ve", "vp"]):
                error_correlation = pearsonr(self.uncertainties[:,i], self.errors[:,i])[0]
                noise_correlation = pearsonr(self.uncertainties[:,i], self.noise_levels)[0]
                self.log(f"test/{p}_uct_error_correlation", error_correlation)
                self.log(f"test/{p}_uct_noise_correlation", noise_correlation)

    def on_test_epoch_start(self):
        # collect all errors, uncertainties, and noise levels to calculate correlations over the entire test set
        self.errors = []
        self.uncertainties = []
        self.noise_levels = []

    def predict_step(self, x):
        self.encoder.eval()
        self.encoder.to(Device)
        out = self.forward(x)
        return out
    
    def predict_batched(self, x, bs=128):
        """
        Evaluates model in batches to avoid memory issues
        Currently not working with MPS backend. Use predict_step instead.
        """
        self.encoder.eval()
        self.encoder.to(Device)
        if self.setup == "mve_pinn":
            self.dt_predictor.to(Device)
            self.dt_predictor.encoder.to(Device)
            self.dt_predictor.encoder.eval()
        with torch.no_grad():
            predictions = {"pred": [], "reconstruction": [], "var": []}
            for i in tqdm(range(0, x.shape[0], bs)):
                out = self.forward(x[i:i+bs])
                params = out["pred"]
                x_hat = out["reconstruction"]
                var = out["var"]
                predictions["pred"].append(params)
                predictions["reconstruction"].append(x_hat)
                predictions["var"].append(var)
        predictions["pred"] = torch.cat(predictions["pred"], dim=0)
        predictions["reconstruction"] = torch.cat(predictions["reconstruction"], dim=0)
        predictions["var"] = torch.cat(predictions["var"], dim=0)
        return predictions

    def visualise(self, datamodule,split, bs=128):
        """
        Visualise predictions and uncertainties
        """
        dataset = {
            "train": datamodule.train_dataset,
            "val": datamodule.val_dataset,
            "test": datamodule.test_dataset
        }[split]
        figs = super().visualise(datamodule=datamodule, split=split, bs=bs)
        if self.mode =="sim":
            if datamodule.ood and self.setup == "ensemble":
                figs["ood"] = plot_ood_class(self.predictions[split], datamodule)
            figs["uct_err_smooth"] = plot_uncertainty_error(self.predictions[split], dataset)
            figs["hist"] = plot_paramdist_uct(self.predictions[split], dataset, datamodule)
            figs["uct_nl"] = plot_uncertainty_noise(self.predictions[split], dataset)
        return figs
    
    def setup_task(self, num_outputs):
        super().setup_task(num_outputs)
        if self.mode == "sim":
            self.train_metrics_uct = MetricsParamsUncertainty(num_outputs=4, prefix="train/")
            self.val_metrics_uct = MetricsParamsUncertainty(num_outputs=4, prefix="val/")
            self.test_metrics_uct = MetricsParamsUncertainty(num_outputs=4, prefix="test/")

    def vis_slice(self, slice_idx=1):
        """
        Visualise the predicted uncertainty in a slice.
        """
        figs, slicedata = super().vis_slice(slice_idx)
        uct = slicedata["var"].cpu().numpy()
        mask = slicedata["mask"]
        shape = mask.shape
        fig, axs = plt.subplots(1, 3, dpi=500)
        width_in_inches = 190 / 25.4
        fig.set_size_inches(width_in_inches, width_in_inches/3)
        for i, axi in enumerate(axs.flat):
            if i > 0:
                j = i + 1
            else:
                j = i
            im = axi.imshow(uct[:,j].reshape(shape)*mask, cmap='inferno')
            axi.set_title(f"{["ke", "dt", "ve", "vp"][j]}")
            axi.axis('off')
            bar = fig.colorbar(im, ax=axi, fraction=0.045, shrink=0.8)
            bar.ax.tick_params(labelsize=6)
        fig.suptitle("Uncertainty", fontsize=15)
        plt.tight_layout()
        figs[f"slice_{slice_idx}_uct"] = fig
        return (figs, slicedata)


