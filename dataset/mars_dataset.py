import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import rasterio
import random

class RealMarsDataset(Dataset):
    def __init__(self, pan_dir, dtm_dir, min_std_threshold=1e-3):
        self.samples = [] # conterrà coppie di percorsi di file (PAN, DTM) 
        self.pan_dir = pan_dir
        self.dtm_dir = dtm_dir

        pan_files = {f: os.path.join(pan_dir, f) for f in os.listdir(pan_dir) if f.endswith(".tif")}
        dtm_files = {f: os.path.join(dtm_dir, f) for f in os.listdir(dtm_dir) if f.endswith(".tif")}
        shared_files = sorted(set(pan_files.keys()) & set(dtm_files.keys()))

        all_dtms = []
        for fname in shared_files:
            with rasterio.open(dtm_files[fname]) as src:
                dtm = src.read(1).astype(np.float32)
                all_dtms.append(dtm.flatten())
                self.samples.append((pan_files[fname], dtm_files[fname]))

        stacked = np.concatenate(all_dtms)
        self.dtm_min = stacked.min()
        self.dtm_max = stacked.max()

        print(f"✅ Dataset loaded correctly.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pan_path, dtm_path = self.samples[idx]

        with rasterio.open(pan_path) as pan_src:
            pan = pan_src.read(1).astype(np.float32)
        with rasterio.open(dtm_path) as dtm_src:
            dtm = dtm_src.read(1).astype(np.float32)

        # normalizzo PAN in [0, 1]
        pan = pan / 255.0

        # min-max scaling DTM (aggiunto 1e-8 per calcoli che toccano 0)
        dtm = (dtm - self.dtm_min) / (self.dtm_max - self.dtm_min + 1e-8)

        # Data augmentation PAN
        if random.random() < 0.5:
            pan = np.fliplr(pan)
            dtm = np.fliplr(dtm)
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            pan = np.rot90(pan, k)
            dtm = np.rot90(dtm, k)
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, pan.shape)
            pan = np.clip(pan + noise, 0.0, 1.0)

        pan_tensor = torch.from_numpy(pan).unsqueeze(0).to(torch.float32)
        dtm_tensor = torch.from_numpy(dtm).unsqueeze(0).to(torch.float32)

        return pan_tensor, dtm_tensor, os.path.basename(pan_path)
