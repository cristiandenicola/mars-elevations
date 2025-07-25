import os
import pandas as pd
import torch
from utils import save_results
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from model.unet import EfficientUNet
from dataset.mars_dataset import RealMarsDataset
from loss.loss import combined_loss
from evaluation.metrics import *
from config import *
from tqdm import tqdm
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

os.makedirs(PRED_SAVE_DIR, exist_ok=True)

# Dataset e dataloader
dataset = RealMarsDataset(CASSIS_PAN, CASSIS_DTM)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modello e ottimizzatore
model = EfficientUNet().to(DEVICE)
loss_fn = combined_loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR, patience=SCHEDULER_PATIENCE
)

# Resume
start_epoch = 0
best_val_loss = float('inf')
no_improve_epochs = 0
log = []

if os.path.exists(LAST_MODEL_SAVE_PATH):
    checkpoint = torch.load(LAST_MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    no_improve_epochs = checkpoint['no_improve_epochs']
    print(f"✔️ Checkpoint caricato. Riprendo da epoca {start_epoch}")

# Funzioni modulari
def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0
    for sample in tqdm(loader, desc="Training", leave=False):
        x = sample["pan"].to(DEVICE)
        y = sample["dtm"].to(DEVICE)

        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def validate_one_epoch(model, loader):
    model.eval()
    val_loss = 0.0
    metric_sums = {
        "rmse": 0.0,
        "mae": 0.0,
        "nmad": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0
    }
    count = 0

    with torch.no_grad():
        for sample in tqdm(loader, desc="Validation", leave=False):
            x = sample["pan"].to(DEVICE)
            y = sample["dtm"].to(DEVICE)

            preds = model(x)
            val_loss += loss_fn(preds, y).item() * x.size(0)

            metrics = compute_metrics(preds.cpu(), y.cpu())
            for k in metric_sums:
                metric_sums[k] += metrics[k] * x.size(0)
            count += x.size(0)

    avg_metrics = {k: v / count for k, v in metric_sums.items()}
    return val_loss / count, avg_metrics

# Training
for epoch in range(start_epoch, EPOCHS):
    print(f"\n🔁 Epoca {epoch + 1}/{EPOCHS}")
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_metrics = validate_one_epoch(model, val_loader)

    print(f"📉 Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    print("📊 Metriche:", {k: round(v, 4) for k, v in val_metrics.items()})

    # Logging
    log.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        **val_metrics
    })
    pd.DataFrame(log).to_csv(LOG_CSV_SAVE_PATH, index=False)

    # Scheduler
    scheduler.step(val_loss)

    # Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'no_improve_epochs': no_improve_epochs,
    }, LAST_MODEL_SAVE_PATH)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
        print("💾 Nuovo best model salvato.")
        save_results.save_predictions(model, val_loader, DEVICE, PRED_SAVE_DIR)
    else:
        no_improve_epochs += 1
        print(f"⏸️ Nessun miglioramento. Epoche senza progresso: {no_improve_epochs}/{EARLY_STOP_PATIENCE}")

    if no_improve_epochs >= EARLY_STOP_PATIENCE:
        print("🛑 Early stopping attivato.")
        break