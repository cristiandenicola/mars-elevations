import os
import torch
import numpy as np
from PIL import Image
from model.unet import EfficientUNet
from dataset.mars_dataset import RealMarsDataset
from config import *
from evaluation.render import render_3d
import matplotlib.pyplot as plt

def split_into_patches(img, patch_size, stride):
    h, w = img.shape
    patches, positions = [], []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            positions.append((i, j))
    return patches, positions, h, w

def stitch_patches(patches, positions, h, w, patch_size):
    stitched = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for patch, (i, j) in zip(patches, positions):
        stitched[i:i+patch_size, j:j+patch_size] += patch
        count[i:i+patch_size, j:j+patch_size] += 1
    return stitched / np.maximum(count, 1)


# ======= LOAD IMAGE ======= 
PAN_IMAGE_PATH = "/Users/cristiandenicola/Documents/data/r2d2 2/CaSSIS_TiffDTMs/MY34_005124_346_1/1/CAS-OTH-MY34_005124_346_1-OPD-03-01-PAN_2.tif"
img_raw = Image.open(PAN_IMAGE_PATH).convert("L")
img_np = np.array(img_raw, dtype=np.float32) / 255.0

# ======= SPLIT =======
patches, positions, h, w = split_into_patches(img_np, PATCH_SIZE, STRIDE)

# ======= LOAD MODEL =======
model = EfficientUNet().to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# ======= PREDICT PATCHES =======
predicted_patches = []
with torch.no_grad():
    for patch in patches:
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        pred = model(patch_tensor).squeeze().cpu().numpy()
        predicted_patches.append(pred)

# ======= STITCH =======
test_dataset = RealMarsDataset(CASSIS_PAN, CASSIS_DTM)

# recupero dtm stats x normalizzazione
dtm_mean = test_dataset.dtm_mean
dtm_std = test_dataset.dtm_std

full_prediction = stitch_patches(predicted_patches, positions, h, w, PATCH_SIZE)
full_prediction = full_prediction * dtm_std + dtm_mean  # denormalizza

# visualize
plt.imshow(full_prediction, cmap="terrain")
plt.title("Predicted Elevation Map")
plt.colorbar()
plt.axis("off")
plt.show()

render_3d(full_prediction)

out_path = os.path.join("output", os.path.basename(PAN_IMAGE_PATH).replace(".tif", "_pred.tif"))
Image.fromarray(full_prediction.astype(np.float32)).save(out_path)
print(f"✅ Predizione salvata in: {out_path}")
