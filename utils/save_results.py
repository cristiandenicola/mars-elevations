import os
import json
import torch
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

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

def save_predictions(model, val_loader, device, save_dir, num_images=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    count = 0
    with torch.no_grad():
        for sample in val_loader:
            pan = sample["pan"].to(device)
            dtm = sample["dtm"].to(device)
            name = sample["name"]

            pred = model(pan)

            for i in range(pan.size(0)):
                if count >= num_images:
                    return
                
                # Estrai le singole immagini
                pan_img = pan[i].squeeze().cpu().numpy()
                gt_img = dtm[i].squeeze().cpu().numpy()
                pred_img = pred[i].squeeze().cpu().numpy()
                file_name = name[i]

                # Scala in [0, 255] per PNG (opzionale: mantieni float per .tif/.npy)
                def to_uint8(x):
                    x = np.clip(x, 0, 1)
                    return (x * 255).astype(np.uint8)

                iio.imwrite(os.path.join(save_dir, f"{count}_pan.png"),  to_uint8(pan_img))
                iio.imwrite(os.path.join(save_dir, f"{count}_gt.png"),   to_uint8(gt_img))
                iio.imwrite(os.path.join(save_dir, f"{count}_pred.png"), to_uint8(pred_img))

                count += 1
