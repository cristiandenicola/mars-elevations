import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import pandas as pd
import torch
from utils import save_results
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from model.swin_unet import EnhancedSwinDepth
from dataset.mars_dataset import RealMarsDataset
from loss.loss import combined_loss_with_perceptual, VGGPerceptualLoss
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

ENHANCED_BATCH_SIZE = max(16, BATCH_SIZE - 4)

train_loader = DataLoader(train_dataset, batch_size=ENHANCED_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=ENHANCED_BATCH_SIZE, shuffle=False)

# Modello enhanced
model = EnhancedSwinDepth()
if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
model = model.to(DEVICE)

# Loss functions
perceptual_loss_fn = VGGPerceptualLoss().to(DEVICE)
loss_fn = combined_loss_with_perceptual(perceptual_loss_fn=perceptual_loss_fn)

# Optimizer con i tuoi parametri ottimizzati
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE,  # Usa il tuo LR (1e-4 è perfetto)
    weight_decay=WEIGHT_DECAY,  # Usa il tuo weight_decay (1e-2 è ottimo)
    eps=1e-8,
    betas=(0.9, 0.999)
)

# Scheduler con i tuoi parametri
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=LR_FACTOR,  # 0.5 dal tuo config
    patience=SCHEDULER_PATIENCE,  # 5 dal tuo config
    min_lr=1e-7
)

# Warm-up scheduler per le prime epoche (con il tuo LR)
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs=3, base_lr=LEARNING_RATE):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

warmup_scheduler = WarmupScheduler(optimizer)

# Resume checkpoint logic (stesso di prima)
start_epoch = 0
best_val_loss = float('inf')
no_improve_epochs = 0
log = []

if os.path.exists(LAST_MODEL_SAVE_PATH):
    checkpoint = torch.load(LAST_MODEL_SAVE_PATH)
    state_dict = checkpoint['model_state_dict']
    
    if list(state_dict.keys())[0].startswith('module.') and not isinstance(model, torch.nn.DataParallel):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    no_improve_epochs = checkpoint['no_improve_epochs']
    print(f"✔️ Checkpoint caricato. Riprendo da epoca {start_epoch}")

scaler = GradScaler('cuda')

# Enhanced training function con multi-scale supervision
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    running_loss = 0.0
    running_main_loss = 0.0
    running_aux_loss = 0.0
    
    for sample in tqdm(loader, desc="Training", leave=False):
        x = sample["pan"].to(DEVICE)
        y = sample["dtm"].to(DEVICE)

        optimizer.zero_grad()
        
        # 1. Forward pass con autocast per mixed precision
        with autocast(device_type='cuda'):
            output = model(x)
            
            # Gestisci multi-scale supervision
            if isinstance(output, tuple):
                main_pred, aux_preds = output
                
                # Main loss
                main_loss = loss_fn(main_pred, y)
                
                # Auxiliary losses
                aux_loss = 0.0
                for i, aux_pred in enumerate(aux_preds):
                    weight = 0.4 * (0.8 ** i)
                    aux_loss += loss_fn(aux_pred, y) * weight
                
                total_loss = main_loss + aux_loss
                running_main_loss += main_loss.item() * x.size(0)
                running_aux_loss += aux_loss.item() * x.size(0)
            else:
                total_loss = loss_fn(output, y)
                running_main_loss += total_loss.item() * x.size(0)
        
        # 2. Backpropagation e aggiornamento dei pesi con lo scaler
        scaler.scale(total_loss).backward()
        
        # Gradient clipping per stabilità
        scaler.unscale_(optimizer) # Da chiamare prima del clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        
        # 3. Aggiornamento dello scaler
        scaler.update()
        
        running_loss += total_loss.item() * x.size(0)
    
    avg_loss = running_loss / len(loader.dataset)
    avg_main_loss = running_main_loss / len(loader.dataset)
    avg_aux_loss = running_aux_loss / len(loader.dataset) if running_aux_loss > 0 else 0
    
    return avg_loss, avg_main_loss, avg_aux_loss

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

            output = model(x)
            
            # In validation usa sempre solo main prediction
            if isinstance(output, tuple):
                preds = output[0]  # Main prediction
            else:
                preds = output
                
            val_loss += loss_fn(preds, y).item() * x.size(0)

            metrics = compute_metrics(preds.cpu(), y.cpu())
            for k in metric_sums:
                metric_sums[k] += metrics[k] * x.size(0)
            count += x.size(0)

    avg_metrics = {k: v / count for k, v in metric_sums.items()}
    return val_loss / count, avg_metrics

# Enhanced training loop
print(f"🚀 Inizio training con modello Enhanced")
print(f"📊 Parametri del modello: {sum(p.numel() for p in model.parameters()):,}")
print(f"📊 Batch size adattato: {ENHANCED_BATCH_SIZE}")

for epoch in range(start_epoch, EPOCHS):
    print(f"\n🔁 Epoca {epoch + 1}/{EPOCHS}")
    
    # Warmup per le prime epoche
    if epoch < 3:
        warmup_scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"🔥 Warmup LR: {current_lr:.2e}")
    
    # Training con multi-scale supervision
    train_loss, main_loss, aux_loss = train_one_epoch(model, train_loader, optimizer, scaler)
    val_loss, val_metrics = validate_one_epoch(model, val_loader)

    # Enhanced logging
    print(f"📉 Train loss: {train_loss:.4f} (main: {main_loss:.4f}, aux: {aux_loss:.4f})")
    print(f"📉 Val loss: {val_loss:.4f}")
    print("📊 Metriche:", {k: round(v, 4) for k, v in val_metrics.items()})
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"📚 Learning rate: {current_lr:.2e}")

    # Enhanced logging
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

    # Scheduler step dopo warmup
    if epoch >= 3:
        scheduler.step(val_loss)

    # Checkpoint con informazioni aggiuntive
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
    
    # Best model logic
    if val_loss < best_val_loss:
        improvement = best_val_loss - val_loss
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
        print(f"💾 Nuovo best model salvato! Miglioramento: {improvement:.6f}")
        save_results.save_predictions(model, val_loader, DEVICE, PRED_SAVE_DIR)
    else:
        no_improve_epochs += 1
        print(f"⏸️ Nessun miglioramento. Epoche senza progresso: {no_improve_epochs}/{EARLY_STOP_PATIENCE}")

    # Early stopping con la tua patience
    if no_improve_epochs >= EARLY_STOP_PATIENCE:  # 15 dal tuo config
        print("🛑 Early stopping attivato.")
        break

print("🏁 Training completato!")
print(f"🏆 Best validation loss: {best_val_loss:.6f}")

# Salva statistiche finali
final_stats = {
    'best_val_loss': best_val_loss,
    'total_epochs': epoch + 1,
    'total_parameters': sum(p.numel() for p in model.parameters()),
    'final_metrics': val_metrics
}

with open(os.path.join(PRED_SAVE_DIR, 'training_stats.json'), 'w') as f:
    import json
    json.dump(final_stats, f, indent=2)