import os
import shutil
import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
from skimage.measure import shannon_entropy

root_dir = "/Users/cristiandenicola/Documents/data/r2d2 2/CaSSIS_TiffDTMs/"
output_dir = "/Users/cristiandenicola/Documents/data/mars_datasets_v2/"

# Parametri di configurazione ottimizzati
patch_size = 256
overlap = 0.5
stride = int(patch_size * (1 - overlap))

# Soglie di qualità più severe
black_threshold = 0.05  # Max 5% di pixel a valore zero
variance_threshold = 50.0  # La varianza deve essere significativa
shannon_entropy_threshold = 2.0  # La patch deve essere informativa

# Prepara la directory di output
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    
os.makedirs(output_dir)
os.makedirs(os.path.join(output_dir, 'DTM'))
os.makedirs(os.path.join(output_dir, 'PAN'))

def extract_high_quality_patches(dtm_path, pan_path, patch_size, stride):
    """
    Estrae patch di alta qualità e senza artefatti di bordo.
    """
    patches = []
    try:
        with rasterio.open(dtm_path) as dtm_src, rasterio.open(pan_path) as pan_src:
            dtm_full = dtm_src.read(1)
            pan_full = pan_src.read(1)

            height, width = dtm_full.shape
            
            # Scorri solo le aree che non creano artefatti di bordo
            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    dtm_patch = dtm_full[y:y+patch_size, x:x+patch_size]
                    pan_patch = pan_full[y:y+patch_size, x:x+patch_size]

                    # Filtri di qualità più severi
                    if np.mean(dtm_patch == 0) > black_threshold or np.mean(pan_patch == 0) > black_threshold:
                        continue

                    if np.var(dtm_patch) < variance_threshold or np.var(pan_patch) < variance_threshold:
                        continue
                        
                    if shannon_entropy(dtm_patch) < shannon_entropy_threshold or shannon_entropy(pan_patch) < shannon_entropy_threshold:
                        continue

                    patches.append(((dtm_patch, x, y), (pan_patch, x, y)))
    
    except Exception as e:
        print(f"Errore durante l'elaborazione dei file {dtm_path} e {pan_path}: {e}")
    
    return patches

print("🚀 Avvio estrazione di patch di alta qualità...")

for subdir in tqdm(os.listdir(root_dir)):
    subdir_path = os.path.join(root_dir, subdir, '1')
    if os.path.isdir(subdir_path):
        dtm_file = pan_file = None

        for file in os.listdir(subdir_path):
            if "-DTM-" in file and file.endswith(".tif"):
                dtm_file = os.path.join(subdir_path, file)
            elif file.endswith("-PAN_1.tif"):
                pan_file = os.path.join(subdir_path, file)

        if dtm_file and pan_file:
            patches = extract_high_quality_patches(dtm_file, pan_file, patch_size, stride)

            for idx, ((dtm_patch, x, y), (pan_patch, _, _)) in enumerate(patches):
                dtm_path = os.path.join(output_dir, 'DTM', f"{subdir}_x{x}_y{y}.tif")
                pan_path = os.path.join(output_dir, 'PAN', f"{subdir}_x{x}_y{y}.tif")

                with rasterio.open(dtm_path, "w", driver="GTiff",
                                   height=patch_size, width=patch_size,
                                   count=1, dtype=dtm_patch.dtype) as dst:
                    dst.write(dtm_patch, 1)

                with rasterio.open(pan_path, "w", driver="GTiff",
                                   height=patch_size, width=patch_size,
                                   count=1, dtype=pan_patch.dtype) as dst:
                    dst.write(pan_patch, 1)

print("\n✅ Creazione del dataset di alta qualità completata!")