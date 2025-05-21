import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import rasterio
import random
from config import * 

class RealMarsDataset(Dataset):
    def __init__(self, pan_dir, dtm_dir):
        self.samples = [] # conterrà coppie di percorsi di file (PAN, DTM) 
        self.pan_dir = pan_dir
        self.dtm_dir = dtm_dir

        pan_files = {f: os.path.join(pan_dir, f) for f in os.listdir(pan_dir) if f.endswith(".tif")}
        dtm_files = {f: os.path.join(dtm_dir, f) for f in os.listdir(dtm_dir) if f.endswith(".tif")}
        shared_files = sorted(set(pan_files.keys()) & set(dtm_files.keys()))

        self.samples = [(pan_files[fname], dtm_files[fname]) for fname in shared_files]
        print(f"✅ Dataset loaded correctly with {len(self.samples)} samples.")

    def read_raster(self, path, nan_override=None, normalize_type=None):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata_val = nan_override if nan_override is not None else src.nodata
            if nodata_val is not None:
                data[data == nodata_val] = np.nan

            if normalize_type == "pan":
                valid_mask = np.isfinite(data) # Identifica i valori non NaN
                if np.any(valid_mask):
                    p2 = np.percentile(data[valid_mask], 2)
                    p98 = np.percentile(data[valid_mask], 98)
                    if p98 - p2 > 0:
                        data = (data - p2) / (p98 - p2) # Normalizza tra p2 e p98
                        data = np.clip(data, 0.0, 1.0)
                    else:
                        data = np.zeros_like(data, dtype=np.float32)
                else:
                    data = np.zeros_like(data, dtype=np.float32)
                data = np.nan_to_num(data, nan=0.0) 

                return data, p2, p98

            elif normalize_type == "dtm":
                min_val = np.nanmin(data)
                max_val = np.nanmax(data)
                if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
                    data = (data - min_val) / (max_val - min_val + 1e-8)
                data = np.nan_to_num(data, nan=0.0)

                return data, min_val, max_val
        
    def augment(self, pan, dtm):
        if random.random() < 0.5:
            pan = np.fliplr(pan).copy()
            dtm = np.fliplr(dtm).copy()
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            pan = np.rot90(pan, k).copy()
            dtm = np.rot90(dtm, k).copy()
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, pan.shape).astype(np.float32)
            pan = np.clip(pan + noise, 0.0, 1.0).copy()

        return pan, dtm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pan_path, dtm_path = self.samples[idx]

        pan, min_val, max_val = self.read_raster(pan_path, nan_override=-32767.0, normalize_type="pan")
        dtm, min_val, max_val = self.read_raster(dtm_path, normalize_type="dtm")

        pan, dtm = self.augment(pan, dtm)

        pan_tensor = torch.from_numpy(pan).unsqueeze(0).float()
        dtm_tensor = torch.from_numpy(dtm).unsqueeze(0).float()

        sample = {
            "pan": pan_tensor,
            "dtm": dtm_tensor,
            "name": os.path.basename(pan_path),
            "min_val": torch.tensor(min_val).float(),
            "max_val": torch.tensor(max_val).float()
        }

        return sample
