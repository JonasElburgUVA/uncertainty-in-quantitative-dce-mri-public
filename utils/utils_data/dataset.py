import torch
import numpy as np
from torch.utils.data import Dataset
import os

class VoxelSet(Dataset):
    # Construct a dataset from an npz object with keys 'params', 'concentrations', 'snr' if mode = 'sim', and 'concentrations' if mode = 'vivo'
    def __init__(self, datapath, mode, split, simmode=None, transform=None, OOD=False):
        self.mode = mode
        mode = f"{mode}/{simmode}" if mode == "sim" else f"{mode}/voxel"
        if OOD:
            mode = f"{mode}_ood"

        datapath = os.path.join(datapath, mode, f"{split}.npz")
        data = np.load(datapath)
        self.transform = transform
        self.split = split
        self.data = data
        if self.mode == 'sim':
            self.params = torch.tensor(data['params']).type(torch.float32)
            self.snr = torch.tensor(data['snr']).type(torch.float32)
        self.ct = torch.tensor(data['concentrations']).type(torch.float32)

    def __getitem__(self, index):
        ct = self.ct[index]
        if self.transform:
            ct = self.transform(ct)
        if self.mode == 'sim':
            return ct, self.params[index], self.snr[index]
        else:
            return ct, ct
        
    def __len__(self):
        return len(self.ct)

class VolumeSet(Dataset):
    def __init__(self, datapath, split, transform=None):
        self.transform = transform
        self.datapath = os.path.join(datapath, f"{split}.npz")
        self.data = np.load(self.datapath)
        self.ct = torch.tensor(self.data['concentrations']).type(torch.float32)
        self.mask = torch.tensor(self.data['mask']).type(torch.int8)

    def __getitem__(self, index):
        ct = self.ct[index]
        if self.transform:
            ct = self.transform(ct)
        return ct
    
    def __len__(self):
        return len(self.ct)
