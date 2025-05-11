import os, sys

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
sys.path.append(os.getcwd())
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .dataset import VoxelSet
import torch

class MyDataModule(LightningDataModule):
    def __init__(self, datapath, mode, param=0, batch_size=32, num_workers=0, simmode="normal", SNR=False, OOD=False):
        super().__init__()
        self.train_dataset = VoxelSet(datapath=datapath,mode=mode,split="train", simmode=simmode, OOD=OOD)
        self.val_dataset = VoxelSet(datapath=datapath,mode=mode,split="val", simmode=simmode, OOD=OOD)
        self.test_dataset = VoxelSet(datapath=datapath,mode=mode,split="test", simmode=simmode, OOD=OOD)

        self.mode = mode
        self.simmode= simmode if mode=="sim" else "vivo"
        self.ood = OOD

        self.batch_size = batch_size
        self.num_workers = num_workers

    def collate_fn_tensordataset(self, batch):
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        if self.mode == "sim":
            snr = torch.stack([item[2] for item in batch])
            return {"input": inputs, "target": targets, "snr": snr}
        return {"input": inputs, "target": targets}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn_tensordataset, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn_tensordataset, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self, shuffle=False):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn_tensordataset, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn_tensordataset, shuffle=False, num_workers=self.num_workers)