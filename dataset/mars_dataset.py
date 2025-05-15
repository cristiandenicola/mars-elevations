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
        self.stats_path = os.path.join(DATA_DIR, "dataset_stats.json")

        pan_files = {f: os.path.join(pan_dir, f) for f in os.listdir(pan_dir) if f.endswith(".tif")}
        dtm_files = {f: os.path.join(dtm_dir, f) for f in os.listdir(dtm_dir) if f.endswith(".tif")}
        shared_files = sorted(set(pan_files.keys()) & set(dtm_files.keys()))

        self.samples = [(pan_files[fname], dtm_files[fname]) for fname in shared_files]

        if os.path.exists(self.stats_path):
            with open(self.stats_path, "r") as f:
                stats = json.load(f)
                self.pan_min = stats["pan_min"]
                self.pan_max = stats["pan_max"]
                self.dtm_min = stats["dtm_min"]
                self.dtm_max = stats["dtm_max"]
            print("Stats loaded from cache.")
        else:
            print("⏳ Computing dataset stats...")
            
            # streaming progressivo (vers concatenate è OOM)
            self.dtm_min = float("inf")
            self.dtm_max = float("-inf")
            self.pan_min = float("inf")
            self.pan_max = float("-inf")


            for pan_path, dtm_path in self.samples:
                with rasterio.open(dtm_path) as src_dtm, rasterio.open(pan_path) as src_pan:
                    dtm = src_dtm.read(1).astype(np.float32)
                    pan = src_pan.read(1).astype(np.float32)

                    self.dtm_min = min(self.dtm_min, float(dtm.min()))
                    self.dtm_max = max(self.dtm_max, float(dtm.max()))
                    self.pan_min = min(self.pan_min, float(pan.min()))
                    self.pan_max = max(self.pan_max, float(pan.max()))

            with open(self.stats_path, "w") as f:
                json.dump({
                    "pan_min": self.pan_min,
                    "pan_max": self.pan_max,
                    "dtm_min": self.dtm_min,
                    "dtm_max": self.dtm_max
                }, f)

            print("✅ Stats computed and saved.")

        print(f"✅ Dataset loaded correctly.")
        print(f"📊 DTM min: {self.dtm_min:.2f}, max: {self.dtm_max:.2f}")
        print(f"📊 PAN min: {self.pan_min:.2f}, max: {self.pan_max:.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        def read_raster(path):
            with rasterio.open(path) as src:
                return src.read(1).astype(np.float32)
            
        pan_path, dtm_path = self.samples[idx]
        pan = read_raster(pan_path)
        dtm = read_raster(dtm_path)

        # min-max scaling PAN
        pan = (pan - self.pan_min) / (self.pan_max - self.pan_min + 1e-8)

        # min-max scaling DTM
        dtm = (dtm - self.dtm_min) / (self.dtm_max - self.dtm_min + 1e-8)

        # Data augmentation PAN
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

        pan_tensor = torch.from_numpy(pan).unsqueeze(0).to(torch.float32)
        dtm_tensor = torch.from_numpy(dtm).unsqueeze(0).to(torch.float32)

        return pan_tensor, dtm_tensor, os.path.basename(pan_path)
