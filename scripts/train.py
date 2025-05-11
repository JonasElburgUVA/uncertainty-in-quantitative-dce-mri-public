import os, sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import ruamel.yaml
import pytorch_lightning as pl
from networks import get_network
from models import get_model
from utils.utils_data import get_dataset 
from utils.utils_data.datamodule import MyDataModule
from utils.utils_torch import Device
import wandb
import matplotlib.pyplot as plt
from models.mcd import MCDropoutWrapper
import os
import json
import argparse
import matplotlib

def main(config, seed, vis):
    matplotlib.use('Agg')
    pl.seed_everything(seed)
    c=config
    ds = c['dataset']
    outpath = f"output/{ds}/{config['model']}_{config['network'].split('_')[0]}_{seed}"
    if config["debug"]:
        outpath += "_debug"
    if config["mcd"]:
        weights_path = f"output/{config['dataset']}/{config['model']}_{config['network'].split('_')[0]}_{seed}/best.ckpt"
        config["weights"] = weights_path
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Could not find the weights file at {weights_path}")
        outpath = config["weights"].split("/")[:-1]
        outpath.append("mcd")
        outpath = "/".join(outpath)
        assert config["epochs"] == 0, "MCDropout is only supported for inference"
        assert config["weights"] is not None, "MCDropout requires a trained model"

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    config["outpath"] = outpath

    # define logger
    if config["logger"] == "wandb":
        s = "mcd_" if config["mcd"] else ""
        logger = pl.loggers.wandb.WandbLogger(project='DCE-MRI',
                                              name=f'{config["model"]}_{config["network"].split('_')[0]}_{config["dataset"]}_{s}{seed}' if config["debug"] == False else f'{config["model"]}_{config["network"]}_{config["dataset"]}_debug')
        wandb.require("core")

    else:
        print("logger unknown or undefined, using default logger. Define your logger in the 'train.py' script if you wish to use a different logge.r")
        logger = None
    
    if logger != None:
        logger.log_hyperparams(config)
        logger.log_hyperparams({"seed": seed})

    # Load baseline model if required
    try:
        p_baseline = f'output/{config["dataset"]}/pinn_ph_dcenet_{seed}/best.ckpt'
        print("succesfully loaded baseline model")
        if not os.path.exists(p_baseline):
            p_baseline = f'output/normal/deterministic/baseline_dcenet_{seed}/best.ckpt'
        config['kwargs']['dt_predictor'] = p_baseline
    except:
        print("Could not load baseline model")
        pass

    # Load network, model, trainer and data
    network = get_network(c['network'])
    model_kwargs = config['kwargs'] if 'kwargs' in config else {}
    model = get_model(model_name=c['model'], 
                      network=network, 
                      weights=config["weights"],
                      mode=config["mode"],
                      loggertrue=True if config['logger'] != None else False,
                      lr=config["lr"], **model_kwargs)
    trainer = pl.Trainer(max_epochs=config["epochs"], 
                         limit_train_batches=config["batch_per_epoch"],
                         limit_val_batches=config["batch_per_epoch"],
                         logger=logger,
                         enable_checkpointing=True,# if not config["debug"] else False,
                         log_every_n_steps=1,
                         accelerator='cpu'if Device == 'cpu' else 'auto',
                         check_val_every_n_epoch=model_kwargs["burnin_epochs"]+1 if "burnin_epochs" in model_kwargs else 1,
                         callbacks=[pl.callbacks.ModelCheckpoint(dirpath=outpath, filename="best", monitor='val_loss', mode='min', verbose=True, enable_version_counter=False), 
                                    pl.callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_epochs'], mode='min', verbose=True)],
                        gradient_clip_val=c['gradient_clip'] if 'gradient_clip' in c else None,
                        # gradient_clip_algorithm="value",
                        num_sanity_val_steps=0,
                        detect_anomaly=False
                        )
    
    # datamodule = MyDataModule(datapath='data/', mode='sim', simmode='normal', batch_size=config['batch_size'])
    datamodule = get_dataset(config['dataset'], batch_size=config['batch_size'])
    # Train and test model
    trainer.fit(model, datamodule)
    if config["mcd"]:
        model = MCDropoutWrapper(model)

    # add best checkpoint to config and save it
    config["weights_out"] = os.path.join(outpath, "best.ckpt")
    with open(os.path.join(outpath, "config.yaml"), "w") as f:
        ruamel.yaml.dump(config, f, ruamel.yaml.RoundTripDumper)

    if config["mcd"]:
        results = trainer.test(model, datamodule)
    else:
        results = trainer.test(model, datamodule, ckpt_path='best' if config["epochs"] > 0 else config["weights_out"])
    # yaml dump results
    with open(os.path.join(outpath, 'results.yaml'), 'w') as f:
        ruamel.yaml.dump(results, f, ruamel.yaml.RoundTripDumper)
    
    print("Training and testing complete. Visualising predictions...")

    # predict and visualise
    try:
        vis_path = os.path.join(outpath, "visualisations")
        os.makedirs(vis_path, exist_ok=True)
        if vis:
            figs = model.visualise(datamodule,split='test', bs=256)# if not config["mcd"] or config["mve_ph"] else 256)

            for k,v in figs.items(): 
                v.savefig(os.path.join(vis_path, f"test_{k}.png"))
        else:
            print("Visualisation disabled. Set 'vis' to True in the config file to enable visualisation. The script can be reran with 0 epochs and the pretrained weights, or you can use eval.py to visualise.")

    except Exception as e:
        print(e)
        print("Error in visualising predictions or logging the visualisation to wandb. Continuing without visualisation...")
    return outpath

if __name__ == "__main__":
    # config = "configs/config.yaml"
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="debug")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--vis", type=bool, default=True)
    args = argparser.parse_args()
    config = os.path.join("configs", f"{args.config}.yaml")
    with open(config, "r") as f:
        config = ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)
    main(config, seed=args.seed, vis=args.vis)