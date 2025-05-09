import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

class RealMarsDataset(Dataset):
    def __init__(self, pan_dir, dtm_dir, min_std_threshold=1e-3):
        self.samples = [] # conterrà coppie di percorsi di file (PAN, DTM) 
        self.min_std_threshold = min_std_threshold # soglia minima per la dev std della pan
        self.pan_dir = pan_dir
        self.dtm_dir = dtm_dir

        pan_files = {f: os.path.join(pan_dir, f) for f in os.listdir(pan_dir) if f.endswith(".tif")}
        dtm_files = {f: os.path.join(dtm_dir, f) for f in os.listdir(dtm_dir) if f.endswith(".tif")}
        

        shared_files = sorted(set(pan_files.keys()) & set(dtm_files.keys()))
        # arrivato qui ho la lista ordinata delle coppie pan-dtm

        self.dtm_stats = []  # temporaneo per calcolo media/std globale DTM

        for fname in shared_files:
            pan_path = pan_files[fname]
            dtm_path = dtm_files[fname]

            pan = Image.open(pan_path).convert("L") # converto in scala di grigi
            pan_arr = np.array(pan, dtype=np.float32) / 255.0 # converto in array numpy e normalizzo [0,1]

            # check if dev std dell'img > min_std_threshold e n pixel max > 0
            # cosi escludo img piatte o tutte nere
            # TODO: chiedere a tia se ha senso questa cosa (fatta perchè sembrano esserci molte img inutili)
            # post controllo > 28148 valid samples found out of 72336
              
            if np.std(pan_arr) > self.min_std_threshold and np.max(pan_arr) > 0:
                self.samples.append((pan_path, dtm_path))
                dtm_arr = np.array(Image.open(dtm_path).convert("L"), dtype=np.float32)
                self.dtm_stats.append(dtm_arr.flatten())

        # calcolo normalizzazione globale DTM
        # calcolo media/dev std di tutte le dtm
        all_dtm = np.concatenate(self.dtm_stats)
        self.dtm_mean = all_dtm.mean()
        self.dtm_std = all_dtm.std()
        del self.dtm_stats

        print(f"✅ Dataset loaded: {len(self.samples)} valid samples found out of {len(shared_files)}")
        print(f"📊 DTM mean: {self.dtm_mean:.2f}, std: {self.dtm_std:.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pan_path, dtm_path = self.samples[idx]

        # TODO: usare tensore torch 
        pan = Image.open(pan_path).convert("L")
        dtm = Image.open(dtm_path).convert("L")

        pan = np.array(pan, dtype=np.float32) / 255.0
        dtm = np.array(dtm, dtype=np.float32)

        # Data augmentation (solo sul PAN)
        # flip orizzontale, flip di 90 gradi (1, 2 o 3 volte), rumore gaussiano con media 0 e deviazione standard 0.01
        # TODO: usare torch.transform 
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

        # normalizzazione DTM
        # DTM viene normalizzato sottraendo la media globale (self.dtm_mean) e dividendo per la deviazione standard globale
        # 1e-8 aggiunto per evitare la divisione per zero nel caso in cui la deviazione standard sia zero.
        dtm = (dtm - self.dtm_mean) / (self.dtm_std + 1e-8)
        dtm = dtm.astype(np.float32)


        pan_tensor = torch.tensor(pan.copy(), dtype=torch.float32).unsqueeze(0)
        dtm_tensor = torch.tensor(dtm.copy(), dtype=torch.float32).unsqueeze(0)


        return pan_tensor, dtm_tensor, os.path.basename(pan_path)
