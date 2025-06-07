import os
import json
import torch
import rasterio
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from config import *

DTM_VIS_MAX_RANGE = GLOBAL_DTM_MAX_ABS + DTM_PREDICTION_MARGIN

# USATA IN test.py
def save_test_results(results_dict, output_dir="test_results", predictions_subdir="predictions", metrics_filename="metrics.json"):
    """
    Salva i risultati del test, incluse le metriche per ogni immagine e, opzionalmente,
    le immagini delle predizioni e dei target.

    Args:
        results_dict (dict): Un dizionario contenente i risultati per ogni immagine,
                             strutturato come {filename: {"metrics": {...}, "timestamp": "..."}}.
        output_dir (str): La directory principale dove salvare i risultati.
        predictions_subdir (str): La sottodirectory per salvare le immagini delle predizioni.
        metrics_filename (str): Il nome del file JSON per salvare le metriche.
    """
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, predictions_subdir)
    os.makedirs(predictions_path, exist_ok=True)
    metrics_file_path = os.path.join(output_dir, metrics_filename)

    with open(metrics_file_path, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"\n💾 Risultati individuali salvati in: {metrics_file_path}")

    print(f"💾 Predizioni e Target (se presenti) salvati in: {predictions_path}")

# USATA IN test.py
def save_prediction_images(output_np, target_np, filename, output_dir="test_results", predictions_subdir="predictions"):
    """
    Salva le immagini della predizione e del target.

    Args:
        output_np (np.ndarray): L'array NumPy della predizione.
        target_np (np.ndarray): L'array NumPy del target.
        filename (str): Il nome del file immagine originale (senza estensione).
        output_dir (str): La directory principale.
        predictions_subdir (str): La sottodirectory per le predizioni.
    """
    predictions_path = os.path.join(output_dir, predictions_subdir)
    os.makedirs(predictions_path, exist_ok=True)
    plt.imsave(os.path.join(predictions_path, f"{filename}_prediction.png"), output_np, cmap='viridis')
    plt.imsave(os.path.join(predictions_path, f"{filename}_target.png"), target_np, cmap='viridis')

def save_predictions(model, dataloader, device, save_dir, num_images=5, fixed_indices=[0, 10, 25, 50, 100]):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    fixed_saved = 0
    random_saved = 0
    total_saved = 0
    max_to_save = len(fixed_indices) + num_images

    # Prendiamo indici random a priori
    dataset_size = len(dataloader.dataset)
    random_indices = set(torch.randperm(dataset_size).tolist()[:num_images])
    fixed_indices = set(fixed_indices)

    with torch.no_grad():
        global_idx = 0
        for batch in dataloader:
            pan = batch["pan"].to(device)
            dtm = batch["dtm"].to(device)
            names = batch["name"]

            preds = model(pan)

            for b in range(pan.size(0)):
                if total_saved >= max_to_save:
                    return

                if global_idx in fixed_indices:
                    label = f"fixed_{global_idx}_{names[b]}"
                    save_single_prediction(pan[b], dtm[b], preds[b], save_dir, label)
                    fixed_saved += 1
                    total_saved += 1

                elif global_idx in random_indices:
                    label = f"random_{global_idx}_{names[b]}"
                    save_single_prediction(pan[b], dtm[b], preds[b], save_dir, label)
                    random_saved += 1
                    total_saved += 1

                global_idx += 1


def save_single_prediction(pan_img, gt, pred, save_dir, name, original_profile=None):
    # original_profile: un dizionario contenente i metadati dell'immagine TIFF originale
    # Se non fornito, il TIFF salvato non avrà georeferenziazione.

    pan_img = pan_img.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    # --- Funzioni di scalatura per visualizzazione PNG (8-bit) ---
    def scale_dtm_for_vis(arr, max_val_for_vis=DTM_VIS_MAX_RANGE):
        normalized_arr = arr / max_val_for_vis
        clipped_arr = np.clip(normalized_arr, 0, 1)
        return (clipped_arr * 255).astype(np.uint8)

    def scale_pan_for_vis(arr, vmin_pan=-2.0, vmax_pan=2.0):
        normalized_arr = (arr - vmin_pan) / (vmax_pan - vmin_pan)
        clipped_arr = np.clip(normalized_arr, 0, 1)
        return (clipped_arr * 255).astype(np.uint8)

    # --- SALVATAGGIO DELLE IMMAGINI PNG (per ispezione visiva rapida) ---
    iio.imwrite(os.path.join(save_dir, f"{name}_pan_viz.png"),  scale_pan_for_vis(pan_img))
    iio.imwrite(os.path.join(save_dir, f"{name}_gt_viz.png"),   scale_dtm_for_vis(gt))
    iio.imwrite(os.path.join(save_dir, f"{name}_pred_viz.png"), scale_dtm_for_vis(pred))

    # --- SALVATAGGIO DEI DATI ORIGINALI FLOAT come GeoTIFF (.tif) ---
    # Creazione di un profilo base per il TIFF. Se original_profile è disponibile, usalo.
    if original_profile:
        # Aggiorna il profilo per il tipo di dati e la dimensione della banda (1 canale)
        # Assicurati che l'altezza e la larghezza nel profilo siano corrette (256x256)
        # e che il dtype sia float32 (o float64 se necessario).
        output_profile = original_profile.copy()
        output_profile.update(
            dtype=rasterio.float32,
            count=1,
            height=pan_img.shape[0], # 256
            width=pan_img.shape[1]  # 256
        )
    else:
        # Profilo di default se non ci sono informazioni georeferenziate
        output_profile = {
            'driver': 'GTiff',
            'height': pan_img.shape[0],
            'width': pan_img.shape[1],
            'count': 1,
            'dtype': rasterio.float32,
            'crs': None, # Nessun sistema di riferimento di coordinate
            'transform': rasterio.transform.from_origin(0, pan_img.shape[0], 1, 1) # Transformazione identity
        }

    # Funzione helper per salvare un singolo array come TIFF
    def save_array_as_tif(data_array, filepath, profile):
        with rasterio.open(filepath, 'w', **profile) as dst:
            dst.write(data_array.astype(rasterio.float32), 1)

    save_array_as_tif(pan_img, os.path.join(save_dir, f"{name}_pan.tif"), output_profile)
    save_array_as_tif(gt, os.path.join(save_dir, f"{name}_gt.tif"), output_profile)
    save_array_as_tif(pred, os.path.join(save_dir, f"{name}_pred.tif"), output_profile)