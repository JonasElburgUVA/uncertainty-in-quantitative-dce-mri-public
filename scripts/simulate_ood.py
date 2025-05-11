import os, sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import numpy as np
import torch
import random

from utils.utils_torch import etofts, Bounds
# from src.utils.globals import Bounds, Device
import matplotlib.pyplot as plt
plt.style.use('classic')
import pickle, json

def simulate_dataset(
        mode: str,
        n: int,
        save: bool = False,
        save_path: str = None,
        cov: np.ndarray = None,
        mean: np.ndarray = None,
        snr_range: list = [1,20],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Simulate a dataset of n samples. For each sample, we create a set of parameters, a matching concentration curve, and an snr value.
    Parameters are sampled following the specified mode. The concentration curve is generated using the etofts function. 
    The snr value is sampled from a uniform distribution between the specified range.

    Mode: str
        'uniform' or 'normal'
        if 'normal', a 4x4 covariance matrix must be provided
    n: int
        number of samples to generate. After filtering out curves, the number of samples may be less than n.
    save: bool
        if True, save the dataset to a file
    cov: np.ndarray
        4x4 covariance matrix. Required if mode is 'normal'
    mean: np.ndarray
        4x1 mean vector. Required if mode is 'normal'
    snr_range: list
        [min, max] snr values to sample from
    """
    N = n
    bounds = Bounds.cpu().numpy()
    if mode == 'uniform':
        n = int(n*1.3)
        params = np.random.uniform(bounds[0], bounds[1], (n, 4))
    elif mode == 'normal':
        if cov is None:
            raise ValueError("Mean and Covariance matrix must be provided for normal mode")
        n *= 4
        params = np.random.multivariate_normal(mean, cov, n)

    # ensure ke=0 if and only if ve = 0
    params[:,0] = np.where(params[:,2] == 0, 0, params[:,0])
    params[:,2] = np.where(params[:,0] == 0, 0, params[:,2])

    # filter out params that are out of bounds
    mask = np.all(params >= bounds[0], axis=1) & np.all(params <= bounds[1], axis=1)
    params = params[mask]
    n = params.shape[0]
    print(f"Number of samples after filtering step 1: {n}")

    params = torch.tensor(params, device=device)
    
    concentrations = etofts(params, batching=True).to(torch.float32).cpu()
    # filter out curves with no peak
    mask = concentrations.max(dim=-1).values > 0.5
    concentrations = concentrations[mask][:N]
    params = params[mask][:N]

    n = concentrations.shape[0]
    print(f"Number of samples after filtering step 2: {n}")


    # add noise to concentration and signal curves
    # SNR = torch.tensor(random.choices(range(snr_range[0],snr_range[1]), k=n))/100
    # noise = np.random.normal(0, torch.mean(concentrations, dim=-1, keepdim=True)[0].repeat(1,80) / SNR.unsqueeze(1).repeat(1,80), concentrations.shape)
    # noise = np.random.normal(0, SNR.unsqueeze(1).repeat(1,80), concentrations.shape)

    NL = torch.tensor(random.choices(range(snr_range[0],snr_range[1]), k=n))/100
    noise = np.random.normal(0, NL.unsqueeze(1).repeat(1,80), concentrations.shape)
    concentrations += noise

    # plot a curve for each unique SNR
    n_subplots = len(NL.unique())
    fig_a, ax = plt.subplots(int(np.ceil(n_subplots/4)), 4, figsize=(10, 10))
    for i, axi in enumerate(ax.flat):
        nl_unique = NL.unique()[i]
        idx = NL == nl_unique
        axi.plot(concentrations[idx][0], label="concentration")
        axi.title.set_text(f"NL: {np.round(nl_unique.item(), 2)}")
        # no xticks
        axi.set_xticks([])

        if i == n_subplots - 1:
            fig_a.tight_layout()
            break
    plt.show()

    params = params.cpu().numpy()
    concentrations = concentrations.cpu().numpy()

    # test_split = np.any([params[:,i] > 1.5*np.mean(params[:,i]) for i in range(4)], axis=0)
    # print(f"Number of OOD samples: {np.sum(test_split)}")
    # train_split = ~test_split
    # val_split = np.random.choice(np.where(train_split)[0], int(0.15*n), replace=False)
    # train_split[val_split] = False

    splits = {
        'train': [0, 0.7],
        'val': [0.7, 0.85],
        'test': [0.85, 1]
    }

    # if OOD:
    #     # mask where all params are > 1.5*mean param
    #     OOD_mask = np.all(params > 1.5*np.mean(params, axis=0), axis=1)
    #     test_params = params[OOD_mask]
    #     test_concentrations = concentrations[OOD_mask]
    #     test_NL = NL[OOD_mask]
    #     params = params[~OOD_mask]
    #     concentrations = concentrations[~OOD_mask]
    #     NL = NL[~OOD_mask]



    config = {
        'mode': mode,
        'n': n,
        'snr_range': snr_range,
        'splits': splits
    }

    splits_data = {}

    if save:
        # save dataset as npz object in 3 splits
        if save_path is None:
            save_path = f"data/sim/{mode}_ood"
        os.makedirs(save_path, exist_ok=True)

        # save config
        with open(f"{save_path}/config.json", "w") as f:
            json.dump(config, f, indent=2)

        for split, size in splits.items():
            params_split = params[int(size[0]*n):int(size[1]*n)]
            concentrations_split = concentrations[int(size[0]*n):int(size[1]*n)]
            NL_split = NL[int(size[0]*n):int(size[1]*n)]
            # if OOD: take out OOD samples from train split
            if split == 'train' or split == 'val':
                # mask where all params are > 1.5*mean param
                OOD_mask = np.any(params_split > (np.mean(params_split, axis=0) + 2*np.std(params_split, axis=0)), axis=1)
                # for the third parameter, we want to exclude the OOD samples of only 1 std above mean
                OOD_mask = OOD_mask | (params_split[:,2] > (np.mean(params_split, axis=0)[2] + np.std(params_split, axis=0)[2]))
                train_params = params_split[~OOD_mask]
                train_concentrations = concentrations_split[~OOD_mask]
                train_NL = NL_split[~OOD_mask]
                params_split = train_params
                concentrations_split = train_concentrations
                NL_split = train_NL

            splits_data[split] = params_split


            print(f"Saving {split} split: {params_split.shape[0]} samples")
            np.savez(f"{save_path}/{split}.npz", params=params_split, concentrations=concentrations_split, snr=NL_split)

        # save figs

        # plot histograms of samples
        param_names = ["ke", "dt", "ve", "vp"]
        fig_b, axs = plt.subplots(1,3, dpi=500)
        width=190 #mm
        width_in_inches = width/25.4
        fig_b.set_size_inches(width_in_inches, width_in_inches/3)

        for i, axi in enumerate(axs.ravel()):
            if i > 0:
                j = i + 1
            else:
                j = i

            axi.hist(splits_data['train'][:,j], bins=100, alpha=0.4, label="train")
            axi.hist(splits_data['test'][:,j], bins=100, alpha=0.5, label="test")
            axi.set_title(param_names[j], fontsize=15)
            axi.legend(fontsize=6)
            # set tick fontsize
            axi.tick_params(axis='both', which='major', labelsize=6)
            # set label fontsize
            axi.set_xlabel("Value", fontsize=10)
            axi.set_ylabel("Frequency", fontsize=10)

        plt.tight_layout()
        fig_b.tight_layout()
        # plt.show()

        fig_a.savefig(os.path.join(save_path, "curves.png"))
        fig_b.savefig(os.path.join(save_path, "hist.png"))

    return params, concentrations, NL

if __name__ == "__main__":
    with open("data/learned_prior/traincov.pkl", "rb") as f:
        cov = pickle.load(f)
    with open("data/learned_prior/trainmean.pkl", "rb") as f:
        mean = pickle.load(f)

    params, concentrations, NL = simulate_dataset('normal', 10000000, save=False, cov=cov, mean=mean, snr_range=[5,25])

