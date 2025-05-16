from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.unet import EfficientUNet
from dataset.mars_dataset import RealMarsDataset
from evaluation.metrics import rmse, mae, nmad, delta_metrics
from utils.visualize import show_prediction
from utils.save_results import *
from evaluation.render import render_3d
from config import *

# Caricamento modello e pesi
model = EfficientUNet().to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# Dataset di test
test_dataset = RealMarsDataset(CASSIS_PAN, CASSIS_DTM)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Dizionario per salvare i risultati per ogni immagine
all_metrics = {}

with torch.no_grad():
    for idx, (image, target, fname) in enumerate(test_loader):
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)

        input_img = image.squeeze().cpu().numpy()
        output_np = output.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()

        # Calcolo metriche
        current_rmse = rmse(output_np, target_np)
        current_mae = mae(output_np, target_np)
        current_nmad = nmad(output_np, target_np)
        d1, d2, d3 = delta_metrics(output_np, target_np)

        metrics_dict = {
            "rmse": current_rmse,
            "mae": current_mae,
            "nmad": current_nmad,
            "delta1": d1,
            "delta2": d2,
            "delta3": d3,
        }

        # Aggiungi le metriche al dizionario con il nome del file come chiave
        filename = fname[0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_metrics[filename] = {"metrics": metrics_dict, "timestamp": timestamp}

        # Output
        print(f"\n🖼️ Prediction {idx + 1} - File: {filename}")
        #print(f"[INPUT] min: {input_img.min():.2f}, max: {input_img.max():.2f}, mean: {input_img.mean():.2f}")
        #print(f"[TARGET] min: {target_np.min():.2f}, max: {target_np.max():.2f}")
        show_prediction(input_img, output_np, target_np)
        # render_3d(output_np) # Opzionale

        # Salvataggio predizione e target come immagini (opzionale)
        save_prediction_images(output_np, target_np, filename)

# Salvataggio di tutti i risultati
save_test_results(all_metrics)

# Risultati finali aggregati
rmse_list = [data["metrics"]["rmse"] for data in all_metrics.values()]
mae_list = [data["metrics"]["mae"] for data in all_metrics.values()]
nmad_list = [data["metrics"]["nmad"] for data in all_metrics.values()]
delta1_list = [data["metrics"]["delta1"] for data in all_metrics.values()]
delta2_list = [data["metrics"]["delta2"] for data in all_metrics.values()]
delta3_list = [data["metrics"]["delta3"] for data in all_metrics.values()]

print("\n📊 Risultati Test Aggregati:")
print(f"  RMSE      : {np.mean(rmse_list):.4f}")
print(f"  MAE       : {np.mean(mae_list):.4f}")
print(f"  NMAD      : {np.mean(nmad_list):.4f}")
print(f"  δ1 (<1.25): {np.mean(delta1_list) * 100:.2f}%")
print(f"  δ2 (<1.25²): {np.mean(delta2_list) * 100:.2f}%")
print(f"  δ3 (<1.25³): {np.mean(delta3_list) * 100:.2f}%")
