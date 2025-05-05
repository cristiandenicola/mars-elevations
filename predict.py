import os
import torch
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from model.unet import EfficientUNet
from evaluate import evaluate
from config import *

PATCH_SIZE = 256
STRIDE = 256

def split_image(image, patch_size, stride):
    h, w = image.shape
    patches = []
    positions = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
            positions.append((i, j))
    return patches, positions

def stitch_predictions(preds, positions, shape):
    out = np.zeros(shape)
    count = np.zeros(shape)
    for pred, (i, j) in zip(preds, positions):
        out[i:i + pred.shape[0], j:j + pred.shape[1]] += pred
        count[i:i + pred.shape[0], j:j + pred.shape[1]] += 1
    
    # Gestione sicura della divisione per evitare NaN
    mask = count > 0
    result = np.zeros_like(out)
    result[mask] = out[mask] / count[mask]
    
    return result

def predict_full_image(image_path):
    model = EfficientUNet().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    img = Image.open(image_path).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    patches, positions = split_image(img_np, PATCH_SIZE, STRIDE)
    preds = []
    
    with torch.no_grad():
        for patch in patches:
            input_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred = model(input_tensor).squeeze().cpu().numpy()
            preds.append(pred)
    
    stitched = stitch_predictions(preds, positions, img_np.shape)
    
    # Gestione di eventuali NaN residui
    stitched = np.nan_to_num(stitched, nan=0.0)
    
    return stitched, img_np

def save_colormap(pred, output_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(pred, cmap="terrain")
    plt.colorbar()
    plt.title("Predicted DTM")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pan", required=True, help="Path to PAN image")
    parser.add_argument("--gt", help="Optional path to ground truth DTM")
    parser.add_argument("--out", default="output/full_pred_dtm.tif", help="Output .tif path")
    parser.add_argument("--out_png", default="output/full_pred_dtm.png", help="Output visualization path")
    
    args = parser.parse_args()
    os.makedirs("output", exist_ok=True)
    
    pred_dtm, input_pan = predict_full_image(args.pan)
    
    # Verifica presenza di NaN
    if np.isnan(pred_dtm).any():
        print("Avviso: NaN rilevati nell'output. Sostituzione con zeri.")
        pred_dtm = np.nan_to_num(pred_dtm, nan=0.0)
    
    Image.fromarray(pred_dtm.astype(np.float32)).save(args.out, format="TIFF")
    save_colormap(pred_dtm, args.out_png)
    
    if args.gt and os.path.exists(args.gt):
        gt_dtm = np.array(Image.open(args.gt)).astype(np.float32)
        
        if gt_dtm.shape != pred_dtm.shape:
            print(f"⚠️ Attenzione: Le dimensioni non corrispondono. GT: {gt_dtm.shape}, Pred: {pred_dtm.shape}")
        
        # Verifica presenza di NaN in ground truth
        if np.isnan(gt_dtm).any():
            print("⚠️ Attenzione: NaN rilevati nel ground truth. Sostituzione con zeri.")
            gt_dtm = np.nan_to_num(gt_dtm, nan=0.0)
        
        evaluate(pred_dtm, gt_dtm)
    else:
        print("⚠️ Ground truth DTM not found or not provided. Skipping evaluation.")