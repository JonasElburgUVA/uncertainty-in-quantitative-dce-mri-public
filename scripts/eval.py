"""
Separate evaluation if you just want to evaluate a trained model. Carefully decide what you want to evaluate, e.g. the model, the slices, the predictions, etc.,
and set the corresponding flags. Full evaluation may take a while, and is usually unnecessary.
"""
import os, sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import ruamel.yaml
from models import get_model
import pytorch_lightning as pl
from utils.utils_torch import Device
from utils.utils_data import get_dataset
from networks import get_network
import argparse
import numpy as np
import torch

def main(args):
    path = args.path
    if args.final:
        fmt = "jpg"
    else:
        fmt = "png"
    config_path = os.path.join(path, "config.yaml")
    with open(config_path) as f:
        c = ruamel.yaml.safe_load(f)

    if c["logger"] == "wandb":
        logger = pl.loggers.wandb.WandbLogger(
            entity='jonas-van-elburg',
            project='DCE-MRI',
            name=f'{c["model"]}_{c["network"].split("_")[0]}_{c["dataset"]}' if c["debug"] == False else f'{c["model"]}_{c["network"].split('_'[0])}_{c["dataset"]}'
        )
        logger=None
    else:
        logger = None

    kwargs= c['kwargs'] if 'kwargs' in c else {}
    # remove burnin epochs from kwargs
    if 'burnin_epochs' in kwargs:
        kwargs.pop('burnin_epochs')

    weights_path = c['weights_out']
    model = get_model(
        model_name=c['model'],
        network=get_network(c['network']),
        weights= weights_path,#os.path.join(path, "best.ckpt"),
        loggertrue=True if c['logger'] != None else False,
        mode=c['mode'],
        mtype=c['model'],
        out_path=path,
        **kwargs
    )
    tester = pl.Trainer(
        max_epochs=0,
        logger=logger,
        accelerator='cpu' if Device.type == 'cpu' else 'auto',
        num_sanity_val_steps=0
    )

    datamodule = get_dataset(c['dataset'], c['batch_size']*4)
    preds_path = os.path.join(path, "predictions")
    vis_path = os.path.join(path, "visualisations")
    os.makedirs(preds_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)

    if args.eval:
        if c["model"] == "pinn_ph":
            print("setting model to inference mode.")
            model.set_inference(True)
        results = tester.test(model, datamodule=datamodule)#, ckpt_path=c['weights'])

        with open(os.path.join(path, "results.yaml"), "w") as f:
            ruamel.yaml.dump(results, f, ruamel.yaml.RoundTripDumper)

    if args.eval_vivo:
        if c["model"] == "pinn_ph":
            print("setting model to inference mode.")
            model.set_inference(True) 
            # clear torch cache
            torch.cuda.empty_cache()
        model.mode = "vivo"
        datamodule = get_dataset("vivo", c['batch_size']*4)
        preds = model.predict_batched(datamodule.test_dataset.ct.to(Device), bs=512)
        inp = datamodule.test_dataset.ct.cpu().numpy()
        mse = np.mean((preds['reconstruction'].cpu().numpy() - inp)**2)
        del inp
        results = {
            "mse": float(mse)
        }
        preds = {k: v.cpu().numpy() for k, v in preds.items()}
        os.makedirs(os.path.join(path, "predictions"), exist_ok=True)  
        np.savez(os.path.join(path, "predictions", "vivo_preds.npz"), **preds)

        with open(os.path.join(path, "results_vivo.yaml"), "w") as f:
            ruamel.yaml.dump(results, f, ruamel.yaml.RoundTripDumper)



    if args.save_preds:
        if c["model"] == "pinn_ph":
            print("setting model to inference mode.")
            model.set_inference(True)
        # save predictions for the test set
        with torch.no_grad():
            preds = model.predict_batched(datamodule.test_dataset.ct.to(Device), bs=512)
            if c["mode"] == "sim":
                gt = datamodule.test_dataset.params.cpu().numpy()
                mse = np.mean((preds['pred'].cpu().numpy() - gt)**2)
                print(f"Mean squared error: {mse}")
            elif c["mode"] == "vivo":
                inp = datamodule.test_dataset.ct.cpu().numpy()
                mse = np.mean((preds['reconstruction'].cpu().numpy() - inp)**2)
                print(f"Mean squared reconstruction error: {mse}")

        preds = {k: v.cpu().numpy() for k, v in preds.items()}
        np.savez(os.path.join(preds_path, "test_preds.npz"), **preds)

        # save gt
        # gt = datamodule.test_dataset.params.cpu().numpy()
        # np.save(os.path.join(preds_path, "test_gt.npy"), gt)

    # save the slice predictions
    if args.eval_slices:
        with torch.no_grad():
            figs1, slice1 = model.vis_slice(slice_idx=1)
            figs2, slice2 = model.vis_slice(slice_idx=2)

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
        np.savez(os.path.join(preds_path, "test_slices.npz"), **slice_data)
    
    if args.visualise_train:
        figs = model.visualise(datamodule, split='train', bs=256)
        for name, fig in figs.items():
            fig.savefig(os.path.join(vis_path, f"train_{name}.{fmt}"), format=fmt, bbox_inches='tight')
        
    if args.visualise_test:
        figs = model.visualise(datamodule, split='test', bs=256)
        for name, fig in figs.items():
            fig.savefig(os.path.join(vis_path, f"test_{name}.{fmt}"), format=fmt, bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False, default="output/ood/mve_snn_dcenet_4")
    parser.add_argument("--final", type=bool, default=False)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--save_preds", type=bool, default=0)
    parser.add_argument("--eval_vivo", type=bool, default=0)
    parser.add_argument("--eval_slices", type=bool, default=0)
    parser.add_argument("--visualise_train", type=bool, default=False)
    parser.add_argument("--visualise_test", type=bool, default=1)
    args = parser.parse_args()
    main(args)
