import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import pandas as pd
import torch
from utils import save_results
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from model.swin_unet import EnhancedSwinDepth
from dataset.mars_dataset import RealMarsDataset
from loss.loss import combined_loss_with_perceptual, VGGPerceptualLoss
from evaluation.metrics import *
from config import *
from tqdm import tqdm
import warnings
from rasterio.errors import NotGeoreferencedWarning
import math
from collections import OrderedDict
import sys
import json

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

os.makedirs(PRED_SAVE_DIR, exist_ok=True)

dataset = RealMarsDataset(CASSIS_PAN, CASSIS_DTM)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = EnhancedSwinDepth(pretrained=True)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(DEVICE)

perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)
loss_fn = combined_loss_with_perceptual(perceptual_loss_fn=perceptual_loss_fn)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.7,
    patience=SCHEDULER_PATIENCE,
    min_lr=1e-8
)

def get_lr_for_epoch(optimizer, epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        lr = (LEARNING_RATE - 1e-6) * ((epoch + 1) / warmup_epochs) + 1e-6
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    else:
        return optimizer.param_groups[0]['lr']

def train_one_epoch(model, loader, optimizer, epoch_num):
    model.train()
    running_loss = 0.0
    running_main_loss = 0.0
    running_aux_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Training E{epoch_num}", leave=False)
    for batch_idx, sample in enumerate(pbar):
        x = sample["pan"].to(DEVICE)
        y = sample["dtm"].to(DEVICE)

        optimizer.zero_grad()
        output = model(x)

        if isinstance(output, tuple):
            main_pred, aux_preds = output
            main_loss = loss_fn(main_pred, y)
            aux_loss = sum(
                loss_fn(aux_pred, y) * (0.2 * (0.5 ** i))
                for i, aux_pred in enumerate(aux_preds)
            )
            total_loss = main_loss + aux_loss
        else:
            main_loss = total_loss = loss_fn(output, y)
            aux_loss = torch.tensor(0.0, device=DEVICE)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Invalid loss at batch {batch_idx}: {total_loss.item()}")
            continue

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += total_loss.item()
        running_main_loss += main_loss.item()
        running_aux_loss += aux_loss.item()
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'main': f'{main_loss.item():.4f}',
            'aux': f'{aux_loss.item():.4f}'
        })

    avg_loss = running_loss / len(loader)
    avg_main_loss = running_main_loss / len(loader)
    avg_aux_loss = running_aux_loss / len(loader) if running_aux_loss > 0 else 0
    
    return avg_loss, avg_main_loss, avg_aux_loss

def validate_one_epoch(model, loader):
    model.eval()
    val_loss = 0.0
    metric_sums = {
        "rmse": 0.0, "mae": 0.0, "nmad": 0.0,
        "delta1": 0.0, "delta2": 0.0, "delta3": 0.0
    }
    count = 0
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(loader, desc="Validation", leave=False)):
            x = sample["pan"].to(DEVICE)
            y = sample["dtm"].to(DEVICE)
            
            output = model(x)
            preds = output[0] if isinstance(output, tuple) else output
            
            loss_value = loss_fn(preds, y)
            
            if torch.isnan(loss_value) or torch.isinf(loss_value):
                continue
            
            val_loss += loss_value.item()
            valid_batches += 1

            preds_cpu = preds.to(torch.float32).cpu()
            y_cpu = y.to(torch.float32).cpu()
            
            metrics = compute_metrics(preds_cpu, y_cpu)

            for k in metric_sums:
                if not math.isnan(metrics[k]) and not math.isinf(metrics[k]):
                    metric_sums[k] += metrics[k]
            count += 1

    if valid_batches == 0:
        print("All validation batches are invalid")
        return float('inf'), {k: float('inf') for k in metric_sums}
    
    avg_metrics = {k: v / count for k, v in metric_sums.items()}
    return val_loss / count, avg_metrics

start_epoch = 0
best_val_loss = float('inf')
no_improve_epochs = 0
log = []

if os.path.exists(LAST_MODEL_SAVE_PATH):
    try:
        print(f"Loading checkpoint from: {LAST_MODEL_SAVE_PATH}")
        checkpoint = torch.load(LAST_MODEL_SAVE_PATH, map_location=DEVICE)
        
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.') and not isinstance(model, torch.nn.DataParallel):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
        log = checkpoint.get('log', [])
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch")

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Batch size: {BATCH_SIZE}")

for epoch in range(start_epoch, EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
    current_lr = get_lr_for_epoch(optimizer, epoch)
    print(f"Learning rate: {current_lr:.2e}")
    if math.isnan(current_lr):
        print("Learning rate is NaN, stopping training")
        break
    
    try:
        train_loss, main_loss, aux_loss = train_one_epoch(model, train_loader, optimizer, epoch + 1)
        val_loss, val_metrics = validate_one_epoch(model, val_loader)
        
    except Exception as e:
        print(f"Error during training at epoch {epoch+1}: {e}")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        val_loss = float('inf')
        val_metrics = {k: float('inf') for k in ["rmse", "mae", "nmad", "delta1", "delta2", "delta3"]}
        continue

    print(f"Train loss: {train_loss:.4f} (main: {main_loss:.4f}, aux: {aux_loss:.4f})")
    print(f"Validation loss: {val_loss:.4f}")
    print("Metrics:", {k: round(v, 4) for k, v in val_metrics.items()})

    log_entry = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_main_loss': main_loss,
        'train_aux_loss': aux_loss,
        'val_loss': val_loss,
        'learning_rate': current_lr,
        **val_metrics
    }
    log.append(log_entry)
    pd.DataFrame(log).to_csv(LOG_CSV_SAVE_PATH, index=False)

    scheduler.step(val_loss)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'no_improve_epochs': no_improve_epochs,
        'log': log
    }
    torch.save(checkpoint, LAST_MODEL_SAVE_PATH)
    
    if val_loss < best_val_loss and not (math.isnan(val_loss) or math.isinf(val_loss)):
        improvement = best_val_loss - val_loss
        best_val_loss = val_loss
        no_improve_epochs = 0
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), BEST_MODEL_SAVE_PATH)
        print(f"New best model saved with improvement: {improvement:.6f}")
        save_results.save_predictions(model, val_loader, DEVICE, PRED_SAVE_DIR)
    else:
        no_improve_epochs += 1
        print(f"No improvement. Epochs without progress: {no_improve_epochs}/{EARLY_STOP_PATIENCE}")

    if no_improve_epochs >= EARLY_STOP_PATIENCE:
        print("Early stopping triggered")
        break

print("Training completed")
print(f"Best validation loss: {best_val_loss:.6f}")

final_stats = {
    'best_val_loss': best_val_loss,
    'total_epochs': epoch + 1,
    'total_parameters': sum(p.numel() for p in model.parameters()),
    'final_metrics': val_metrics
}

with open(os.path.join(PRED_SAVE_DIR, 'training_stats.json'), 'w') as f:
    json.dump(final_stats, f, indent=2)