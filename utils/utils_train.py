from torchmetrics import (
    MetricCollection,
    MeanSquaredError)
from torchmetrics.metric import Metric
    # MeanAbsoluteError,
    # R2Score)
# from src.lightning.metrics import (
#     NormalizedRootMeanSquaredError, 
#     GaussianNLL,
#     SNRCorrelation,
#     ErrorCorrelation,
#     MSE_per_param)
from torch import Tensor, tensor
import torch
from torchmetrics.functional.regression.mse import _mean_squared_error_compute, _mean_squared_error_update

class Metrics(MetricCollection):
    def __init__(self, num_outputs, prefix):
        super().__init__({
            "curve/MSE": MeanSquaredError(squared=True, num_outputs=num_outputs),
            "curve/NRMSE": NormalizedRootMeanSquaredError(num_outputs=num_outputs, norm_by_mean=True)
        }, prefix=prefix)

class MetricsParams(MetricCollection):
    def __init__(self, num_outputs, prefix):
        super().__init__({
            "param/MSE": MeanSquaredError(squared=True, num_outputs=4),
            "param/MSE_ke": MSE_per_param(squared=True, num_outputs=1, param=0),
            "param/MSE_dt": MSE_per_param(squared=True, num_outputs=1, param=1),
            "param/MSE_ve": MSE_per_param(squared=True, num_outputs=1, param=2),
            "param/MSE_vp": MSE_per_param(squared=True, num_outputs=1, param=3),
        }, prefix=prefix)

class MetricsParamsUncertainty(MetricCollection):
    def __init__(self, num_outputs, prefix):
        super().__init__({
            "param/GNLL": GaussianNLL(num_outputs=3),
            "param/GNLL_ke": GaussianNLL(num_outputs=1, param=0),
            # "param/GNLL_dt": GaussianNLL(num_outputs=1, param=1),
            "param/GNLL_ve": GaussianNLL(num_outputs=1, param=2),
            "param/GNLL_vp": GaussianNLL(num_outputs=1, param=3)
        }, prefix=prefix)


class MSE_per_param(MeanSquaredError):
    def __init__(
            self,
            squared: bool = True,
            num_outputs: int = 1,
            param: int = 0,
            **kwargs,
    ):
        super().__init__(squared=squared, num_outputs=num_outputs, **kwargs)
        self.param = param

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.param != None:
            preds = preds[:,self.param]
            target = target[:,self.param]
        super().update(preds, target)

class NormalizedRootMeanSquaredError(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    sum_squared_error: Tensor
    total: Tensor

    def __init__(
            self,
            num_outputs: int=1,
            norm_by_mean: bool = False,
            param = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.squared = True
        self.norm_by_mean = norm_by_mean
        self.param = param

        if not (isinstance(num_outputs, int) and num_outputs > 0):
            raise ValueError(f"Expected num_outputs to be a positive integer but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("nrmse", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.param != None:
            preds = preds[:,self.param]
            target = target[:,self.param]

        if self.norm_by_mean:
            # normalize each datapoint by its mean
            normalizer = target.mean(dim=1)
            normalizer[normalizer < 1e-3] = 1e-3
            nrmse = torch.sum(
                        torch.sqrt(
                            torch.sum((preds-target)**2, dim=-1) / normalizer
                        ), dim=0)

        else:
            # normalize each parameter by its mean
            normalizer = target.mean(dim=0).unsqueeze(0)
            nrmse = torch.sum(torch.sqrt((preds-target)**2 / normalizer), dim=0)

        num_obs = target.shape[0]

        self.nrmse += nrmse
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean squared error over state."""
        # if len(self.nrmse.shape) > 0:
        #     self.nrmse = self.nrmse.mean()
        with torch.no_grad():
            return _mean_squared_error_compute(self.nrmse, self.total, squared=self.squared)

class GaussianNLL(Metric):
    r"""
    Computes the Gaussian Negative Log Likelihood (NLL) loss.
    """
    def __init__(
        self,
        num_outputs=1,
        param=None,
    ) -> None:
        super().__init__()
        self.add_state("nll", default=torch.zeros(num_outputs), dist_reduce_fx="mean")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.num_outputs = num_outputs
        self.param = param

    def update(self, mean: Tensor, var: Tensor, target: Tensor) -> None:
        """Update state with mean, std and target tensors."""
        if self.param != None:
            mean = mean[:,self.param]
            var = var[:,self.param] if len(var.squeeze().shape) > 1 else var.squeeze()
            target = target[:,self.param] 
        elif self.param == None:
            mean = mean[:,(0,2,3)]
            target = target[:,(0,2,3)]
            var = var[:,(0,2,3)]
        # var min to 1e-5
        var = torch.clamp(var, min=1e-6)
        nll = 0.5 * (torch.log(var) + ((target - mean) ** 2) / (var))
        num_obs = target.shape[0]
        if nll.isnan().sum() > 0:
            raise ValueError("NLL is NaN")
        nll = torch.mean(nll, dim=0)
     
        self.nll += nll
        self.total += 1

    def compute(self) -> Tensor:
        """Compute mean squared error over state."""
        out = self.nll / self.total
        if out.isnan().sum() > 0:
            raise ValueError("NLL is NaN")  
        return self.nll / self.total
    
