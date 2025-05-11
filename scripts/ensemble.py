"""
Script to evaluate an ensemble model. Use in combination with an ensemble config file, after training N models. Compatible with all uncertainty quantifying models,
including MVE, PINN_PH, and MVE_PINN.
"""
import os, sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import ruamel.yaml
import pytorch_lightning as pl
import wandb
from models import get_model
from utils.utils_data import get_dataset
from models import get_ensemble
from networks import get_network
import argparse
import numpy as np
from utils.utils_torch import Device
import torch


def main(config, args):
    fmt="png"
    pl.seed_everything(config["seed"])
    # define logger
    paths = [os.path.join("output",p) for p in config["paths"]]
    name = "_".join(config["paths"][0].split("_")[:-1]+["ens"])
    if config["logger"] == "wandb":
        logger = pl.loggers.wandb.WandbLogger(entity='jonas-van-elburg', project='DCE-MRI', 
                                              name=f'{name}')
        wandb.require("core")
    else:
        print("logger unknown or undefined, using default logger")
        logger = None

    paths = [os.path.join("output",p) for p in config["paths"]]
    n = len(paths)
    cfgs = [ruamel.yaml.load(open(os.path.join(p, "config.yaml"), "r"), Loader=ruamel.yaml.Loader) for p in paths]
    cfgs = list(cfgs)
    models = [get_model(cfg["model"], 
                        network=get_network(cfg["network"]), 
                        weights=os.path.join(paths[i], "best.ckpt"), 
                        mode=cfg["mode"], 
                        loggertrue=True if config['logger'] != None else False,
                        lr=cfg["lr"],
                        **cfg['kwargs']) for i,cfg in enumerate(cfgs)]
    
    ensemble = get_ensemble(models)
    dm = get_dataset(config['dataset'], batch_size=1024)#MyDataModule(datapath='data/', mode='sim', simmode='normal', batch_size=256)
    # dump config
    os.makedirs(os.path.join("output", name), exist_ok=True)
    with open(os.path.join("output", name, "config.yaml"), "w") as f:
        ruamel.yaml.dump(config, f, ruamel.yaml.RoundTripDumper)
    
    if args.eval:
        tester = pl.Trainer(logger=logger)
        results = tester.test(ensemble, dm)
        with open(os.path.join("output", name, "results.yaml"), "w") as f:
            ruamel.yaml.dump(results, f, ruamel.yaml.RoundTripDumper)
    pred_path = os.path.join("output", name, "predictions")
    os.makedirs(pred_path, exist_ok=True)
    if args.save_preds:
 
        with torch.no_grad():
            pred = ensemble.predict_batched(dm.test_dataset.ct.to(Device), bs=2048)
        pred = {k: v.detach().cpu().numpy() for k,v in pred.items()}
        np.savez(os.path.join(pred_path, "test_preds.npz"), **pred)

    vis_path = os.path.join("output", name, "visualisations")
    os.makedirs(vis_path, exist_ok=True)

    if args.eval_vivo:
        tester = pl.Trainer(logger=logger)
        dm_vivo = get_dataset("vivo", batch_size=1024)
        ensemble.mode = "vivo"
        results = tester.test(ensemble, dm_vivo)
        with open(os.path.join("output", name, "results_vivo.yaml"), "w") as f:
            ruamel.yaml.dump(results, f, ruamel.yaml.RoundTripDumper)
        with torch.no_grad():
            pred = ensemble.predict_batched(dm_vivo.test_dataset.ct.to(Device), bs=2048)
        pred = {k: v.detach().cpu().numpy() for k,v in pred.items()}
        np.savez(os.path.join(pred_path, "vivo_preds.npz"), **pred)


    if args.vis_slices:
        with torch.no_grad():
            figs1, slice1 = ensemble.vis_slice(slice_idx=1)
            figs2, slice2 = ensemble.vis_slice(slice_idx=2)

        for name, fig in figs1.items():
            fig.savefig(os.path.join(vis_path, f"test_slice1_{name}.{fmt}"), format=fmt, bbox_inches='tight', dpi=500)
        for name, fig in figs2.items():
            fig.savefig(os.path.join(vis_path, f"test_slice2_{name}.{fmt}"), format=fmt, bbox_inches='tight', dpi=500)

        slice_data = {
            "slice1": slice1,
            "slice2": slice2
        }
        for k, v in slice_data.items():
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    slice_data[k][kk] = vv.cpu().numpy()
        np.savez(os.path.join(pred_path, "test_slices.npz"), **slice_data)

    if args.vis:
        with torch.no_grad():
            # figs = ensemble.visualise(dm, split='train', bs=2048)    
            # for name, fig in figs.items():
            #     fig.savefig(os.path.join(vis_path, f"train_{name}.png"), bbox_inches='tight')

            figs = ensemble.visualise(dm, split='test', bs=2048)
            for name, fig in figs.items():
                fig.savefig(os.path.join(vis_path, f"test_{name}.png"), bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file', default='configs/normal/ensemble_snn.yaml')
    parser.add_argument('--save_preds', type=bool, help='save predictions', default=0)
    parser.add_argument('--eval', type=bool, help='evaluate model', default=0)
    parser.add_argument('--eval_vivo', type=bool, help='evaluate model on vivo data', default=0)
    parser.add_argument('--vis_slices', type=bool, help='visualise slices', default=1)
    parser.add_argument('--vis', type=bool, help='visualise', default=0)
    args = parser.parse_args()
    config = args.config
    with open(config, "r") as f:
        config = ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)
    main(config, args)
