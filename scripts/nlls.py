"""
Non Linear Least Squares
"""
import os, sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from pytorch_lightning import seed_everything   
import pickle, json
from utils.utils_data.datamodule import MyDataModule
from utils.utils_train import Metrics, MetricsParams, MetricsParamsUncertainty
from utils.utils_torch import aifPopPMB, etofts
from utils.utils_numpy import fit_tofts_model
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


def main(args):
    device = "cpu"
    seed_everything(0)
    # load initial guess
    X0 = pickle.load(open("data/learned_prior/trainmean.pkl", "rb"))
    Timepoints = torch.arange(0, 80, 1).float() * 4 / 60
    Bounds = torch.tensor([[0, 0, 0, 0], [3, 2, 1, 1]])
    batch_size = 32

    dataset = args.mode
    if args.mode == "sim":
        dataset = args.simmode
    # set up output directory
    output_dir = f"output/{dataset}/nlls"
    os.makedirs(output_dir, exist_ok=True)
    dm = MyDataModule(datapath="data", mode=args.mode, batch_size=batch_size, SNR=True if args.mode == "sim" else False, simmode="normal")
    if args.eval:
        split="test"
        test_loader = dm.test_dataloader(shuffle=False)
        if not args.preload:
            metrics = Metrics(num_outputs=80, prefix="test/")
            if dm.mode == "sim":
                metrics_params = MetricsParams(num_outputs=4, prefix="test/")
                metrics_uct = MetricsParamsUncertainty(num_outputs=4, prefix="test/")
            X = dm.test_dataset.ct
            pred_params = torch.empty((X.shape[0], 4))
            recs = torch.empty((X.shape[0], 80))
            ucts = torch.empty((X.shape[0], 4))

            for i, dp in enumerate(tqdm(test_loader)):
                aif = aifPopPMB()
                x = dp["input"].to(device)
                if dm.mode == "sim":
                    p = dp["target"].to(device)
                n = x.shape[0]
                end = min(i*batch_size+n, X.shape[0])
                # lstsq loop
                params, pcov = fit_tofts_model(x.cpu().numpy(), Timepoints.cpu().numpy(), aif, model='Cosine4', X0=X0, bounds=Bounds.cpu().numpy(), jobs=64)
                params = torch.tensor(params, dtype=torch.float32).to(device)
                pcov = torch.tensor(pcov, dtype=torch.float32).to(device)
                # get the diagonals of the covariance matrices with shape (N,4,4) -> (N,4)
                uct = torch.diagonal(pcov, dim1=-2, dim2=-1).to(device)

                if len(params.shape) < 2:
                    params = params.unsqueeze(0)
                x_hat = etofts(params)
                metrics.update(x_hat, x)
                if dm.mode == "sim":
                    metrics_params.update(params, p)
                    metrics_uct.update(params, uct, p)
                
                pred_params[i*batch_size:end] = params
                recs[i*batch_size:end] = x_hat
                ucts[i*batch_size:end] = uct

            # save results
            if args.save:
                np.savez(os.path.join(output_dir, f"{split}_preds.npz"), reconstructions=recs.cpu().numpy(), pred=pred_params.cpu().numpy(), var=ucts.cpu().numpy())

            results = {
                "rec": metrics.compute()}
            
            if dm.mode == "sim":
                results.update({
                "params": metrics_params.compute() if dm.mode == "sim" else None,
                "uct": metrics_uct.compute() if dm.mode == "sim" else None
            })

            results["rec"]["test/curve/MSE"] = results["rec"]["test/curve/MSE"].mean()
            results["rec"]["test/curve/NRMSE"] = results["rec"]["test/curve/NRMSE"].mean()

            if dm.mode == "sim":
                pred_params = pred_params.cpu().numpy()
                y_test = dm.test_dataset.params.cpu().numpy()
                errors = np.abs(pred_params - y_test)
                uct_error_corr ={}
                for i, p in enumerate(["ke", "dt", "ve", "vp"]):
                    uct_error_corr[p] = pearsonr(errors[:,i], ucts[:,i])
                results["uct-error correlation"] = uct_error_corr



            for k,v in results.items():
                if isinstance(v, dict):
                    for kk,vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            results[k][kk] = vv.mean().item()
            with open(os.path.join(output_dir, f"results_{split}.json"), "w") as f:
                json.dump(results, f, indent=4)
        else:
            results = json.load(open(os.path.join(output_dir, "results_test.json"), "r"))
            pred = pickle.load(open(os.path.join(output_dir, f"results_test.pkl"), "rb"))
            pred_params = pred["params"]
            pred_uct = pred["var"]
            y_test = dm.test_dataset.params.cpu().numpy()

            errors = np.abs(pred_params - y_test)
            uct_error_corr ={}
            # uct_nl_corr = {}
            for i, p in enumerate(["ke", "dt", "ve", "vp"]):
                uct_error_corr[p] = pearsonr(errors[:,i], pred_uct[:,i])
                # uct_nl_corr[p] = pearsonr(pred_params[:,i], pred_uct[:,i])

            results["uct-error correlation"] = uct_error_corr
            # results.update(uct_nl_corr)

            json.dump(results, open(os.path.join(output_dir, f"results_{split}.json"), "w"), indent=4)

    # plot
    if args.vis:
        if dm.mode == "sim":
            param_names = ["ke", "dt", "ve", "vp"]
            fig, axs = plt.subplots(2,2, figsize=(15,10))
            for i in range(4):
                axs[i//2, i%2].scatter(y_test[:,i], pred_params[:,i], alpha=0.1, s=8)
                axs[i//2, i%2].set_xlabel("True")
                axs[i//2, i%2].set_ylabel("Predicted")
                axs[i//2, i%2].set_title(f"Parameter {param_names[i]}")
                # plot x=y between min and max of true values
                min_ = y_test[:,i].min()
                max_ = y_test[:,i].max()
                axs[i//2, i%2].plot([min_, max_], [min_, max_], "r--")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "params.png"))
            plt.close()
        elif dm.mode == "vivo":
            slices = {}
            for slice_id in range(1,3):
                name = f"slice{slice_id}"
                ct, _, mask = np.load(f"data/vivo/slice{slice_id}.npz").values()
                flat_ct = ct.reshape(-1, 80)
                N = flat_ct.shape[0]
                out = np.zeros_like(flat_ct)
                flat_ct_masked = np.where(np.expand_dims(mask.reshape(-1),-1).repeat(axis=-1, repeats=80), flat_ct, out)

                # do lstsq fit 
                aif = aifPopPMB()
                params, pcov = [], []
                for batch in tqdm(range(0, flat_ct_masked.shape[0], batch_size)):
                    params_, pcov_ = fit_tofts_model(flat_ct_masked[batch:batch+batch_size], Timepoints.cpu().numpy(), aif, model='Cosine4', X0=X0, bounds=Bounds.cpu().numpy(), jobs=64)
                    params.append(params_)
                    pcov.append(pcov_)
                # params, pcov = fit_tofts_model(flat_ct_masked, Timepoints.cpu().numpy(), aif, model='Cosine4', X0=X0, bounds=Bounds.cpu().numpy(), jobs=64)
                # fill 'out' with params at the masked locations
                out = np.zeros((N,4))
                params = np.vstack(params)
                pcov = np.vstack(pcov)
                out = np.where(np.expand_dims(mask.reshape(-1),-1).repeat(axis=-1, repeats=4), params, out)
                params = out
                uct = np.zeros((N,4))
                uct = np.where(np.expand_dims(mask.reshape(-1),-1).repeat(axis=-1, repeats=4), np.diagonal(pcov, axis1=-2, axis2=-1), uct)
                uct = uct.reshape(mask.shape + (4,))
                rec = etofts(torch.tensor(params))
                rec_error = np.abs(rec - flat_ct).square().mean(axis=-1)
                rec_error = rec_error.reshape(mask.shape)
                rec_m = rec.mean(axis=-1).reshape(mask.shape)

                width_in_inches = 190 / 25.4
                height_in_inches = 190 / (25.4*2)
                fig_a, ax = plt.subplots(1,2)
                fig_a.set_size_inches(width_in_inches, height_in_inches)
                im = ax[0].imshow(rec_m*mask, cmap="gray")
                ax[0].axis("off")
                ax[0].set_title("Reconstruction")
                bar = fig_a.colorbar(im, ax=ax[0], fraction=0.045, shrink=0.8)
                bar.ax.tick_params(labelsize=6)
                im = ax[1].imshow(rec_error*mask, cmap="inferno", vmax=2.0)
                ax[1].set_title("Error")
                ax[1].axis("off")
                bar = fig_a.colorbar(im, ax=ax[1], fraction=0.045, shrink=0.8)
                bar.ax.tick_params(labelsize=6)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"slice{slice_id}.png"))
                

                fig_b, ax = plt.subplots(1,3,dpi=500)
                height_in_inches = 190 / (25.4*3)
                fig_b.set_size_inches(width_in_inches, height_in_inches)
                for i, axi in enumerate(ax.ravel()):
                    if i > 0:
                        j = i + 1
                    else:
                        j = i
                    im = axi.imshow(params[:,j].reshape(mask.shape)*mask, cmap='inferno')
                    axi.set_title(f"{['ke', 'dt', 've', 'vp'][j]}")
                    axi.axis('off')
                    bar = fig_b.colorbar(im, ax=axi, fraction=0.045, shrink=0.8)
                    bar.ax.tick_params(labelsize=6)

                fig_b.suptitle("Parameters", fontsize=15)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"params_slice{slice_id}.png"))


                fig_c, axs = plt.subplots(1, 3, dpi=500)
                width_in_inches = 190 / 25.4
                fig_c.set_size_inches(width_in_inches, width_in_inches/3)
                for i, axi in enumerate(axs.flat):
                    if i > 0:
                        j = i + 1
                    else:
                        j = i
                    im = axi.imshow(uct[:,:,j]*mask, cmap='inferno', vmax=np.percentile(uct[:,:,j], 95))
                    axi.set_title(f"{["ke", "dt", "ve", "vp"][j]}")
                    axi.axis('off')
                    bar = fig_c.colorbar(im, ax=axi, fraction=0.045, shrink=0.8)
                    bar.ax.tick_params(labelsize=6)

                fig_c.suptitle("Uncertainty", fontsize=15)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"uct_slice{slice_id}.png"))

                # save the slice results
                slices[name] = {"reconstruction": rec, "pred": params, "var": uct}
                # np.savez(os.path.join(output_dir, f"slice_results_{slice_id}.npz"), reconstruction=rec, pred=params, var=uct)
            # save the slice results
            np.savez(os.path.join(output_dir, "slice_results.npz"), **slices)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="vivo")
    parser.add_argument("--simmode", type=str, default="normal")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--preload", type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--vis", type=bool, default=True)
    args = parser.parse_args()
    main(args)


