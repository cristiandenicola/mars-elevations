import torch
import numpy as np
from torch.utils.data import DataLoader
from model.unet import EfficientUNet
from dataset.mars_dataset import RealMarsDataset
from evaluation.metrics import rmse, mae, nmad
from utils.visualize import show_prediction
from evaluation.render import render_3d
from config import *

# upload modello e pesi
model = EfficientUNet().to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# data load
test_dataset = RealMarsDataset(CASSIS_PAN, CASSIS_DTM)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# recupero dtm stats x normalizzazione
dtm_mean = test_dataset.dtm_mean
dtm_std = test_dataset.dtm_std

rmse_list, mae_list, nmad_list = [], [], []

with torch.no_grad():
    for idx, (image, target, fname) in enumerate(test_loader):
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)

        input_img = image.squeeze().cpu().numpy()
        output_np = output.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()

        # denormalizzo predizione utilizzando la media e la dev std del DTM calcolate durante la creazione del dataset
        # riporta i valori della predizione alla scala originale del DTM

        # idem per il target
        output_np = output_np * dtm_std + dtm_mean
        target_np = target_np * dtm_std + dtm_mean

        # calcolo metriche
        rmse_list.append(rmse(output_np, target_np))
        mae_list.append(mae(output_np, target_np))
        nmad_list.append(nmad(output_np, target_np))

        # plot show
        print(f"\n🖼️ Showing prediction {idx+1} - File: {fname}")
        print(f"[DEBUG] input min: {input_img.min():.2f}, max: {input_img.max():.2f}, mean: {input_img.mean():.2f}")
        print("GT min/max:", target_np.min(), target_np.max())
        show_prediction(input_img, output_np, target_np)
        render_3d(output_np)

# metriche finali
print("\n📊 Test Results:")
print(f"  RMSE: {np.mean(rmse_list):.4f}")
print(f"  MAE : {np.mean(mae_list):.4f}")
print(f"  NMAD: {np.mean(nmad_list):.4f}")
