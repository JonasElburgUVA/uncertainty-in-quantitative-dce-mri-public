from pytorch_lightning import LightningModule
import torch
import abc
from utils.utils_torch import Device
from utils.utils_train import Metrics, MetricsParams
from functools import partial
from utils.utils_vis import plot_predictions_regression, plot_error_noise
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from networks import get_network
import os

"""
this file contains the abstract bass class for all models.
Abstract methods which need to be implemented in the child classes:
- forward
- _get_loss
- training_step
- validation_step
- test_step
"""
class Base(LightningModule):
    def __init__(self, 
                 network=get_network("fc"),
                 lr=1e-4,
                 loggertrue=False,
                 mode="sim",
                 weights=None,
                 mtype="base",
                 out_path=None,
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.mtype = mtype
        self.out_path = out_path
        if not self.mtype == "ensemble":
            self.encoder = network.to(Device)
            self.lr = lr
            self.save_hyperparameters(ignore=["network"])
            self.num_params = 4#self.encoder(torch.zeros(1,80, device=Device)).shape[1]
            self.setup_task(num_outputs=80)
            self.loggertrue = loggertrue
            self.setup = "base"
            self.configure_optimizers()
            if weights != None:
                self.load_state_dict(torch.load(weights, map_location=Device, weights_only=True)['state_dict'])
            self.encoder.to(Device)
        self.predictions = {}


    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def _get_loss(self, batch):
        pass

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abc.abstractmethod
    def test_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        self.log_dict(self._prepare_metrics_dict(self.train_metrics.compute()), on_epoch=True)
        self.train_metrics.reset()
        if not self.mode == "vivo":
            self.log_dict(self._prepare_metrics_dict(self.train_metrics_params.compute()), on_epoch=True)
            self.train_metrics_params.reset()

    def on_validation_epoch_end(self):
        """
        Logs the validation metrics at the end of the epoch, and performs LR decrease if necessary.
        """
        metrics = self._prepare_metrics_dict(self.val_metrics.compute())
        self.log_dict(metrics, on_epoch=True)
        self.val_metrics.reset()
        if not self.mode == "vivo":
            metrics_params = self._prepare_metrics_dict(self.val_metrics_params.compute())
            self.log_dict(metrics_params, on_epoch=True)
            self.val_metrics_params.reset()
        # get the last Lr
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr", lr)
        # check if lr decreased
        if lr < self.lr:
            self.lr = lr 
            # load best checkpoint
            self.load_state_dict(torch.load(self.trainer.checkpoint_callback.best_model_path, weights_only=True, map_location=Device)['state_dict'])
            print(f"Loaded best model with lr: {lr*10} at the end of epoch {self.current_epoch}")


    def on_test_epoch_end(self):
        metrics = self._prepare_metrics_dict(self.test_metrics.compute())
        self.log_dict(metrics, on_epoch=True)
        self.test_metrics.reset()
        if not self.mode == "vivo":
            metrics_params = self._prepare_metrics_dict(self.test_metrics_params.compute())
            self.log_dict(metrics_params, on_epoch=True)
            self.test_metrics_params.reset()


    def predict_step(self, X, batch_idx=0, dataloader_idx=None):
        """
        Predicts the output of the model given an input X.
        """
        try:
            if isinstance(X, dict):
                X = X['input']
            with torch.no_grad():
                out = self.forward(X)
        except RuntimeError:
            print("RuntimeError in predict_step, try using predict_batched(X, batchsize) instead")
            x_hat = torch.zeros(1,80)
            params = torch.zeros(1, self.num_params)
            out = {"reconstruction": x_hat, "pred": params}
        return out
    
    def predict_batched(self, X, bs):
        """
        Predicts the output of the model given an input X in batches.
        """
        self.encoder.eval()
        self.encoder.to(Device)
        dummy = self.forward(torch.zeros(1,80, device=Device))
        predictions = {k: [] for k in dummy.keys()}
        for i in tqdm(range(0, X.shape[0], bs)):
            out = self.forward(X[i:i+bs])
            for k in out.keys():
                predictions[k].append(out[k])
        for k,v in predictions.items():
            if len(v) > 0:
                predictions[k] = torch.cat(v, dim=0)


        return predictions

    def configure_optimizers(self, lr=None):
        if lr != None:
            self.lr = lr
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) #self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler_config = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'val_loss',
            'strict' : False
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
    
    def re_init(self, mode):
        """
        Re-initializes the model to account for changes in set-up when evaluating on different datasets.
        """
        self.mode = mode
        self.num_params = self.encoder(torch.zeros(1,80, device=Device)).shape[1]
        self.setup_task(num_outputs=80)

    def setup_task(self, num_outputs):
        # Initializes the metrics for the model. Add/Remove metrics in the Metrics class in train_utils.py.
        self.train_metrics = Metrics(num_outputs, prefix="train/")
        self.val_metrics = Metrics(num_outputs, prefix="val/")
        self.test_metrics = Metrics(num_outputs, prefix="test/")
        if not self.mode == "vivo":
            self.train_metrics_params = MetricsParams(self.num_params, prefix="train/")
            self.val_metrics_params = MetricsParams(self.num_params, prefix="val/")
            self.test_metrics_params = MetricsParams(self.num_params, prefix="test/")

    def _prepare_metrics_dict(self, metrics):
        for k,v in metrics.items():
            if len(v.squeeze().shape) > 0:
                metrics[k] = v.mean()
        return metrics
    
    def visualise(self, datamodule, split, bs=128):
        """
        Visualizes the accuracy of predictions of the model on the test set.
        """
        dataset = {
            "train": datamodule.train_dataset,
            "val": datamodule.val_dataset,
            "test": datamodule.test_dataset
        }[split]
        x = dataset.ct.to(Device)[:20000]
        if self.mtype != "ensemble":
            self.encoder.to(Device)
        if self.mode == "sim":
            with torch.no_grad():
                if Device.type != "mps":
                    predictions = self.predict_batched(x, bs)
                else:
                    predictions = self.predict_step(x)
            self.predictions[split] = predictions
        # show fig
            figs = {
                "accuracy": plot_predictions_regression(predictions, dataset, uct=False),
                "err_nl": plot_error_noise(predictions, dataset),
            }
        else:
            figs = {}
        return figs
    
    def vis_slice(self, slice_idx=1):
        """
        Visualizes the CT image, the error in the CT image and the predicted parameters for a given slice.
        """
        ct, _, mask = np.load(f"data/vivo/slice{slice_idx}.npz").values()
        ct = torch.tensor(ct).to(Device).float()
        shape = mask.shape
        if Device.type != "mps":
            predictions = self.predict_batched(ct.view(-1,80), 256)
        else:
            predictions = self.predict_step(ct.view(-1,80))
        ct_mean = ct.cpu().numpy().mean(axis=-1)
        ct_error = (predictions["reconstruction"] - ct.view(-1,80)).abs().mean(dim=-1)
        fig_a, ax = plt.subplots(1,2, dpi=500)
        width_in_inches = 190 / 25.4
        height_in_inches = 190 / (25.4*2)
        fig_a.set_size_inches(width_in_inches, height_in_inches)
        im = ax[0].imshow(ct_mean.reshape(shape)*mask, cmap='inferno', vmax=np.percentile(ct_mean, 99))
        ax[0].axis('off')
        ax[0].set_title("CT", fontsize=15)
        bar = fig_a.colorbar(im, ax=ax[0], fraction=0.045, shrink=0.8)
        bar.ax.tick_params(labelsize=6)
        im = ax[1].imshow(ct_error.cpu().numpy().reshape(shape)*mask, cmap='hot', vmax=np.percentile(ct_error.cpu().numpy(), 99))
        ax[1].axis('off')
        ax[1].set_title("CT error", fontsize=15)
        bar = fig_a.colorbar(im, ax=ax[1], fraction=0.045, shrink=0.8)
        bar.ax.tick_params(labelsize=6)
        plt.tight_layout()
        params = predictions["pred"].cpu().numpy()
        fig_b, ax = plt.subplots(1,3, dpi=500)
        height_in_inches = 190 / (25.4*3)
        fig_b.set_size_inches(width_in_inches, height_in_inches)
        for i, axi in enumerate(ax.ravel()):
            if i > 0:
                j = i+1
            else:
                j = i
            im = axi.imshow(params[:,j].reshape(shape)*mask, cmap='inferno')
            axi.axis('off')
            axi.set_title(["ke", "dt", "ve", "vp"][j])
            bar = fig_b.colorbar(im, ax=axi, fraction=0.045, shrink=0.8)
            bar.ax.tick_params(labelsize=6)
        fig_b.suptitle("Predicted parameters", fontsize=15)
        plt.tight_layout()
        slicedata = predictions
        slicedata["ct"] = ct
        slicedata["mask"] = mask
        return ({f"slice_{slice_idx}_ct": fig_a, f"slice_{slice_idx}_params": fig_b}, slicedata)




