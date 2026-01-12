import os
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
import random
from config import *
import cv2

class RealMarsDataset(Dataset):
    def __init__(self, pan_dir, dtm_dir):
        self.samples = []
        self.pan_dir = pan_dir
        self.dtm_dir = dtm_dir

        pan_files = {f: os.path.join(pan_dir, f) for f in os.listdir(pan_dir) if f.endswith(".tif")}
        dtm_files = {f: os.path.join(dtm_dir, f) for f in os.listdir(dtm_dir) if f.endswith(".tif")}
        shared_files = sorted(set(pan_files.keys()) & set(dtm_files.keys()))

        self.samples = [(pan_files[fname], dtm_files[fname]) for fname in shared_files]
        print(f"Dataset loaded correctly with {len(self.samples)} samples")

    def read_raster_raw(self, path, nan_override=None):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata_val = nan_override if nan_override is not None else src.nodata
            if nodata_val is not None:
                data[data == nodata_val] = np.nan
            return data

    def augment(self, pan_raw, dtm_raw):
        pan = pan_raw.astype(np.float32)
        dtm = dtm_raw.astype(np.float32)

        # --- 1. Trasformazioni Spaziali 

        # Flip Orizzontale
        if random.random() < 0.5:
            pan = np.fliplr(pan).copy()
            dtm = np.fliplr(dtm).copy()

        # Rotazione a 90/180/270 gradi
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            pan = np.rot90(pan, k).copy()
            dtm = np.rot90(dtm, k).copy()

        # Flip Verticale
        if random.random() < 0.3:
            pan = np.flipud(pan).copy()
            dtm = np.flipud(dtm).copy()

        # --- 2. Trasformazioni di Intensità (applicate SOLO a PAN) ---

        # Rumore Gaussiano
        if random.random() < 0.3:
            noise_std = random.uniform(100, 150)
            noise = np.random.normal(0, noise_std, pan.shape).astype(np.float32)
            pan = pan + noise
            pan = np.clip(pan, -32768, 32767).copy()

        # Sfocatura Gaussiana
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            pan = cv2.GaussianBlur(pan, (ksize, ksize), 0).copy()

        # Variazione di Luminosità/Contrasto
        if random.random() < 0.4:
            alpha = random.uniform(0.8, 1.1)
            beta = random.uniform(-150, 150)
            pan = pan * alpha + beta
            pan = np.clip(pan, -32768, 32767).copy()

        return pan, dtm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pan_path, dtm_path = self.samples[idx]

        pan_raw = self.read_raster_raw(pan_path, nan_override=-32767.0)
        dtm_raw = self.read_raster_raw(dtm_path)

        # 2. DATA ARGUMENTATION
        pan_augmented, dtm_augmented = self.augment(pan_raw, dtm_raw)

        valid_mask = ~np.isnan(dtm_augmented)
        
        # Se non ci sono pixel validi, restituisci un tensore zero
        if not np.any(valid_mask):
            pan_tensor = torch.zeros(1, pan_raw.shape[0], pan_raw.shape[1], dtype=torch.float32)
            dtm_tensor = torch.zeros(1, dtm_raw.shape[0], dtm_raw.shape[1], dtype=torch.float32)
            return {"pan": pan_tensor, "dtm": dtm_tensor, "min_val": 0.0, "max_val": 0.0, "name": os.path.basename(pan_path)}

        # Normalizzazione PAN
        pan_normalized = (pan_augmented - GLOBAL_PAN_MEAN) / GLOBAL_PAN_STD

        # Normalizzazione DTM
        dtm_min = np.nanmin(dtm_augmented[valid_mask])
        dtm_max = np.nanmax(dtm_augmented[valid_mask])
        dtm_augmented[valid_mask] = (dtm_augmented[valid_mask] - dtm_min) / (dtm_max - dtm_min + 1e-8)
        dtm_augmented[valid_mask] = dtm_augmented[valid_mask] * TARGET_DTM_NORMALIZATION_RANGE

        # Conversione dei NaN in 0 DOPO la normalizzazione
        pan_augmented = np.nan_to_num(pan_normalized, nan=0.0)
        dtm_augmented = np.nan_to_num(dtm_augmented, nan=0.0)

        pan_tensor = torch.from_numpy(pan_augmented).unsqueeze(0).float()
        dtm_tensor = torch.from_numpy(dtm_augmented).unsqueeze(0).float()

        sample = {
            "pan": pan_tensor,
            "dtm": dtm_tensor,
            "name": os.path.basename(pan_path),
            "min_val": torch.tensor(dtm_min).float(),
            "max_val": torch.tensor(dtm_max).float(),
        }

        return sample