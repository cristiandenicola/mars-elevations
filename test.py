import torch
import numpy as np
import pandas as pd

# === CONFIG ===
GT_PATH = "/home/cdenicola/my_datasets/DTM"  # path DTM GT
PRED_PATHS = {
    "EnhancedSwinDepth": "model/swin_unet.py",
    "EfficientUNet": "model/unetB5.py",
    "EfficientUNetB4": "model/unetB4.py",
}
ELEVATION_RANGE = (0, 600)
BIN_SIZE = 50
SAMPLE_LIMIT = 2000

# === LOAD DATA ===
def load_tensor(path):
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict) and "pred" in t:  # compatibilità
        t = t["pred"]
    return t.squeeze().detach().cpu().numpy().astype(np.float32)

# Carica subset di file
from pathlib import Path
gt_files = sorted(Path(GT_PATH).glob("*.pth"))[:SAMPLE_LIMIT]

predictions = {name: [] for name in PRED_PATHS}
ground_truths = []

for gt_file in gt_files:
    gt = load_tensor(gt_file)
    ground_truths.append(gt)
    for name, folder in PRED_PATHS.items():
        pred_file = Path(folder) / gt_file.name
        if pred_file.exists():
            pred = load_tensor(pred_file)
            predictions[name].append(pred)

# Stack arrays
gt_all = np.concatenate([g.flatten() for g in ground_truths])
preds_all = {name: np.concatenate([p.flatten() for p in preds]) for name, preds in predictions.items()}

# === SCALE DETECTION ===
for name, pred in preds_all.items():
    if np.max(pred) <= 1.5:  # sembra normalizzato
        print(f"[INFO] {name} sembra normalizzato — scaling ×650")
        preds_all[name] = pred * 650.0
    else:
        print(f"[INFO] {name} già in metri (max={np.max(pred):.2f})")

# === ABSOLUTE ERROR PER RANGE ===
bins = np.arange(ELEVATION_RANGE[0], ELEVATION_RANGE[1] + BIN_SIZE, BIN_SIZE)
results = []

for i in range(len(bins) - 1):
    mask = (gt_all >= bins[i]) & (gt_all < bins[i + 1])
    if np.sum(mask) == 0:
        continue
    row = {"Range (m)": f"[{bins[i]}, {bins[i+1]})"}
    for name, pred in preds_all.items():
        abs_error = np.abs(pred[mask] - gt_all[mask])
        mean_error = np.mean(abs_error)
        row[name] = mean_error
    results.append(row)

df = pd.DataFrame(results)
print(df)
