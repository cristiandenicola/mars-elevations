import os
import shutil

import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy.stats import mode
from skimage.measure import shannon_entropy

# Define paths
root_dir = "/Users/cristiandenicola/Documents/data/r2d2 2/CaSSIS_TiffDTMs/"
output_dir = "/Users/cristiandenicola/Documents/data/mars_datasets/"

# Config
patch_size = 256
overlap = 0.5  # 50% overlap

black_threshold = 0.2  # 20% black pixels

# flatness
variance_threshold = 3.0 # + basso + severo
range_threshold = 2.0 # + basso + severo
std_threshold = 0.8 # + basso + severo - scarta quelli che hanno val medio vicino a val, quindi con pochi valori diff

stride = int(patch_size * (1 - overlap))

# Prepare output
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    
os.makedirs(output_dir)
os.makedirs(os.path.join(output_dir, 'DTM'))
os.makedirs(os.path.join(output_dir, 'PAN'))

def pad_image(image, patch_size):
    height, width = image.shape
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    return image

def dominant_value_ratio(patch):
    values, counts = np.unique(patch, return_counts=True)
    return np.max(counts) / patch.size

def extract_valid__patches(dtm_path, pan_path, patch_size, stride):
    with rasterio.open(dtm_path) as dtm_src, rasterio.open(pan_path) as pan_src:
        dtm = pad_image(dtm_src.read(1), patch_size)
        pan = pad_image(pan_src.read(1), patch_size)

        height, width = dtm.shape
        patches = []

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                dtm_patch = dtm[y:y+patch_size, x:x+patch_size]
                pan_patch = pan[y:y+patch_size, x:x+patch_size]

                # Check black ratio on both
                if np.mean(dtm_patch == 0) > black_threshold or np.mean(pan_patch == 0) > black_threshold:
                    continue

                # Check flatness (variance, range, and standard deviation) on both
                if np.var(dtm_patch) < variance_threshold or np.var(pan_patch) < variance_threshold or \
                   np.ptp(dtm_patch) < range_threshold or np.ptp(pan_patch) < range_threshold or \
                   np.std(dtm_patch) < std_threshold or np.std(pan_patch) < std_threshold:
                    continue

                # check how much the patch is covered by a single dominant value
                if dominant_value_ratio(dtm_patch) > 0.95 or dominant_value_ratio(pan_patch) > 0.95:
                    continue
                
                # check how many unique values the patch has (very few = flat)
                if len(np.unique(dtm_patch)) < 13 or len(np.unique(pan_patch)) < 13:
                    continue

                # entropy check
                if shannon_entropy(dtm_patch) < 1.3 or shannon_entropy(pan_patch) < 1.3:
                    continue

                patches.append(((dtm_patch, x, y), (pan_patch, x, y)))

        return patches
    
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
            patches = extract_valid__patches(dtm_file, pan_file, patch_size, stride)

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

print("✅ Dataset creation completed successfully!")
