import os
import numpy as np
import rasterio
import sys
from torch.utils.data import DataLoader # Per caricare il dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

from dataset.mars_dataset import RealMarsDataset
from config import *

def calculate_stats_for_pan(pan_dir, nan_override=-32767.0):
    sum_x = 0.0
    sum_x2 = 0.0
    count = 0
    
    pan_files = [os.path.join(pan_dir, f) for f in os.listdir(pan_dir) if f.endswith(".tif")]

    print(f"Calcolo statistiche globali per {len(pan_files)} immagini PAN (memoria efficiente)...")
    for i, pan_path in enumerate(pan_files):
        with rasterio.open(pan_path) as src:
            data = src.read(1).astype(np.float32)
            if nan_override is not None and src.nodata is not None:
                data[data == src.nodata] = np.nan
            elif nan_override is not None:
                data[data == nan_override] = np.nan

            valid_data = data[np.isfinite(data)]
            
            if valid_data.size > 0:
                # Aggiorna le somme e il conteggio
                sum_x += np.sum(valid_data)
                sum_x2 += np.sum(valid_data**2)
                count += valid_data.size
        
        if (i + 1) % 100 == 0:
            print(f"  Elaborate {i+1}/{len(pan_files)} immagini PAN. Pixel processati: {count}")

    if count == 0:
        global_mean = 0.0
        global_std = 1.0
        print("Nessun pixel PAN valido trovato, utilizzando valori di default.")
    else:
        global_mean = sum_x / count
        variance = (sum_x2 / count) - (global_mean ** 2)
        global_std = np.sqrt(max(0, variance))

    print(f"Media PAN globale: {global_mean}")
    print(f"Deviazione standard PAN globale: {global_std}")
    return global_mean, global_std

def calculate_max_dtm(dtm_dir):
    """Trova il valore massimo assoluto tra tutti i DTM del dataset."""
    overall_max_dtm = -float('inf') # Inizializza con un valore molto piccolo
    dtm_files = [os.path.join(dtm_dir, f) for f in os.listdir(dtm_dir) if f.endswith(".tif")]

    print(f"Calcolo DTM Max globale per {len(dtm_files)} immagini DTM...")
    for i, dtm_path in enumerate(dtm_files):
        with rasterio.open(dtm_path) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                 data[data == src.nodata] = np.nan
            
            current_max = np.nanmax(data)
            if np.isfinite(current_max) and current_max > overall_max_dtm:
                overall_max_dtm = current_max
        
        if (i + 1) % 100 == 0:
            print(f"  Elaborate {i+1}/{len(dtm_files)} immagini DTM.")

    print(f"Valore DTM massimo globale (assoluto): {overall_max_dtm}")
    return overall_max_dtm

if __name__ == "__main__":
    CONFIG_FILE_PATH = os.path.join(project_root, 'config.py')

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Errore: File di configurazione '{CONFIG_FILE_PATH}' non trovato.")
        print("Assicurati che il file esista o crea un config.py di base.")
        exit() 

    if not os.path.exists(CASSIS_PAN) or not os.path.exists(CASSIS_DTM):
        print("Errore: Percorsi di training PAN o DTM non trovati. Si prega di adattare i percorsi.")
        exit()

    # Calcola statistiche PAN
    global_pan_mean, global_pan_std = calculate_stats_for_pan(CASSIS_PAN)

    # Calcola DTM Max assoluto
    global_dtm_max_abs = calculate_max_dtm(CASSIS_DTM)

    # Definizione del range di normalizzazione DTM
    target_dtm_norm_range = global_dtm_max_abs

    # --- Aggiorna il contenuto del file config.py ---
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            lines = f.readlines()

        new_values = {
            "GLOBAL_PAN_MEAN": f"GLOBAL_PAN_MEAN = {global_pan_mean:.6f}\n",
            "GLOBAL_PAN_STD": f"GLOBAL_PAN_STD = {global_pan_std:.6f}\n",
            "GLOBAL_DTM_MAX_ABS": f"GLOBAL_DTM_MAX_ABS = {global_dtm_max_abs:.6f}\n",
            "TARGET_DTM_NORMALIZATION_RANGE": f"TARGET_DTM_NORMALIZATION_RANGE = {target_dtm_norm_range:.6f}\n",
        }

        updated_lines = []
        found_keys = {key: False for key in new_values.keys()}

        for line in lines:
            updated = False
            for key, new_line_content in new_values.items():
                if line.strip().startswith(key):
                    updated_lines.append(new_line_content)
                    found_keys[key] = True
                    updated = True
                    break # Passa alla prossima riga originale
            if not updated:
                updated_lines.append(line)
        
        for key, new_line_content in new_values.items():
            if not found_keys[key]:
                inserted = False
                for i, line in enumerate(updated_lines):
                    if "# Dataset parameters" in line:
                        updated_lines.insert(i + 1, new_line_content)
                        inserted = True
                        break
                if not inserted:
                    updated_lines.append(new_line_content)


        with open(CONFIG_FILE_PATH, 'w') as f:
            f.writelines(updated_lines)

        print(f"\n✅ Valori globali aggiornati con successo in {CONFIG_FILE_PATH}")
        print("Assicurati di rigenerare il modello se è stato addestrato con valori diversi.")

    except Exception as e:
        print(f"❌ Errore durante l'aggiornamento di {CONFIG_FILE_PATH}: {e}")
        print("Si prega di copiare i valori manualmente per sicurezza:")
        print(f"GLOBAL_PAN_MEAN = {global_pan_mean:.6f}")
        print(f"GLOBAL_PAN_STD = {global_pan_std:.6f}")
        print(f"GLOBAL_DTM_MAX_ABS = {global_dtm_max_abs:.6f}")
        print(f"TARGET_DTM_NORMALIZATION_RANGE = {target_dtm_norm_range:.6f}")