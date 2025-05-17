import os
import time
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from model.unet import EfficientUNet
from dataset.mars_dataset import RealMarsDataset
from loss.loss import combined_loss
from evaluation import rmse, mae, delta_metrics
from config import *

SAVE_IMG_EVERY = 5          # salva immagini ogni N epoche
NUM_SAMPLES_TO_SAVE = 3     # quante coppie per epoca
IMG_SAVE_DIR = "pred_vs_gt" # cartella di output
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

def save_predictions(preds, targets, epoch, batch_idx):
    """Salva NUM_SAMPLES_TO_SAVE coppie pred/gt come PNG."""
    preds = preds.detach().cpu().squeeze(1).numpy()
    targets = targets.detach().cpu().squeeze(1).numpy()

    for i in range(min(NUM_SAMPLES_TO_SAVE, preds.shape[0])):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(targets[i], cmap="viridis")
        axes[0].set_title("GT DTM")
        axes[0].axis("off")

        axes[1].imshow(preds[i], cmap="viridis")
        axes[1].set_title("Pred DTM")
        axes[1].axis("off")

        fname = f"epoch{epoch+1:03d}_batch{batch_idx:04d}_sample{i}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_SAVE_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

# Seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# split dataset into train e validation
full_dataset = RealMarsDataset(CASSIS_PAN, CASSIS_DTM)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modello e ottimizzatore
model = EfficientUNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

# Early stopping
best_loss = float("inf")
no_improve_epochs = 0
EARLY_STOP_PATIENCE = 10

# Resume checkpoint
if os.path.exists(LAST_MODEL_SAVE_PATH):
    print("⚠️  Resuming from last checkpoint...")
    model.load_state_dict(torch.load(LAST_MODEL_SAVE_PATH))

# Header CSV
if not os.path.exists(LOG_CSV_SAVE_PATH):
    with open(LOG_CSV_SAVE_PATH, "w") as f:
        f.write("epoch,train_loss,val_loss,mae,rmse,delta1,delta2,delta3,time\n")

# Train loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    for batch_idx, (images, targets, _) in enumerate(train_loader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        preds = model(images)
        loss = combined_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)

    # Validazione
    model.eval()
    val_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for val_images, val_targets, _ in val_loader:
            val_images, val_targets = val_images.to(DEVICE), val_targets.to(DEVICE)
            val_preds = model(val_images)
            loss = combined_loss(val_preds, val_targets)
            val_loss += loss.item()

            all_preds.append(val_preds)
            all_targets.append(val_targets)

            if (epoch + 1) % SAVE_IMG_EVERY == 0:
                save_predictions(val_preds, val_targets, epoch, batch_idx=0)
                break

    avg_val_loss = val_loss / len(val_loader)
    preds_tensor = torch.cat(all_preds, dim=0).squeeze(1).cpu().numpy()
    targets_tensor = torch.cat(all_targets, dim=0).squeeze(1).cpu().numpy()

    val_mae = mae(preds_tensor, targets_tensor)
    val_rmse = rmse(preds_tensor, targets_tensor)
    deltas = delta_metrics(torch.tensor(preds_tensor), torch.tensor(targets_tensor))

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - MAE: {val_mae:.4f} - RMSE: {val_rmse:.4f} - δ1: {deltas['delta1']:.3f} - δ2: {deltas['delta2']:.3f} - δ3: {deltas['delta3']:.3f} - Time: {epoch_time:.2f}s")

    # Logging CSV
    with open(LOG_CSV_SAVE_PATH, "a") as f:
        f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_mae:.4f},{val_rmse:.4f},{deltas['delta1']:.3f},{deltas['delta2']:.3f},{deltas['delta3']:.3f},{epoch_time:.2f}\n")

    scheduler.step(avg_val_loss)

    # Early stopping su best val loss
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print("🛑 Early stopping.")
            break

    # Save ultimo modello
    torch.save(model.state_dict(), LAST_MODEL_SAVE_PATH)

print("Training completo ✅")
