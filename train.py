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
from evaluation.metrics import *
from config import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Settings
SAVE_IMG_EVERY = 1  # Save images every N epochs
NUM_SAMPLES_TO_SAVE = 3  # How many pairs per epoch
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter("runs/mars_experiment")

def save_predictions(preds, targets, epoch, writer):
    """Saves a representative sample of predictions as PNGs and logs to TensorBoard."""
    preds = preds.detach().cpu().squeeze(1).numpy()
    targets = targets.detach().cpu().squeeze(1).numpy()

    num_samples = min(NUM_SAMPLES_TO_SAVE, preds.shape[0])
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))

    # Se num_samples == 1, axes ha shape (2,), altrimenti (num_samples, 2)
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)  # porta axes a shape (1, 2)

    for i in range(num_samples):
        axes[i, 0].imshow(targets[i], cmap="viridis")
        axes[i, 0].set_title("GT DTM")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(preds[i], cmap="viridis")
        axes[i, 1].set_title("Pred DTM")
        axes[i, 1].axis("off")

    fname = f"epoch{epoch + 1:03d}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_SAVE_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)

    writer.add_image(f"Epoch {epoch + 1}/GT", torch.tensor(targets[0]).unsqueeze(0), epoch)
    writer.add_image(f"Epoch {epoch + 1}/Pred", torch.tensor(preds[0]).unsqueeze(0), epoch)


if __name__ == '__main__':
    # Seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # split dataset into train e validation
    full_dataset = RealMarsDataset(CASSIS_PAN, CASSIS_DTM)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    # Modello e ottimizzatore
    model = EfficientUNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Early stopping
    best_loss = float("inf")
    no_improve_epochs = 0
    EARLY_STOP_PATIENCE = 10

    # Resume checkpoint
    if os.path.exists(LAST_MODEL_SAVE_PATH):
        print("⚠️  Resuming from last checkpoint...")
        checkpoint = torch.load(LAST_MODEL_SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        no_improve_epochs = checkpoint['no_improve_epochs']
    else:
        epoch_start = 0

    # Header CSV
    if not os.path.exists(LOG_CSV_SAVE_PATH):
        with open(LOG_CSV_SAVE_PATH, "w") as f:
            f.write("epoch,train_loss,val_loss,mae,rmse,nmad,delta1,delta2,delta3,time\n")

    # Training Loop
    for epoch in range(epoch_start, EPOCHS):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for batch_idx, (images, targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(images)
            loss = combined_loss(preds, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Log batch time
            batch_time = time.time() - start_time
            writer.add_scalar("Train/Batch_Time", batch_time, epoch * len(train_loader) + batch_idx)

            # for name, param in model.named_parameters():
            #    if "conv" in name and param.grad is not None:
            #        if param.grad.numel() > 0:
            #            writer.add_histogram(f"Gradients/{name}", param.grad.cpu().numpy(), epoch * len(train_loader) + batch_idx)
            #        else:
            #            print(f"Warning: Gradiente vuoto per {name} al batch {batch_idx}, epoca {epoch + 1}")

        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)

        # Validazione
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for val_images, val_targets, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                val_images, val_targets = val_images.to(DEVICE), val_targets.to(DEVICE)
                val_preds = model(val_images)
                loss = combined_loss(val_preds, val_targets)
                val_loss += loss.item()

                print(f"Val Targets Min: {val_targets.min().item()}, Max: {val_targets.max().item()}") # Aggiungi questo
                print(f"Val Preds Min: {val_preds.min().item()}, Max: {val_preds.max().item()}")   # Assicurati di avere anche questo

                # Log min/max values
                writer.add_scalar("Validation/Pred_Min", val_preds.min().item(), epoch * len(val_loader))
                writer.add_scalar("Validation/Pred_Max", val_preds.max().item(), epoch * len(val_loader))
                writer.add_scalar("Validation/Target_Min", val_targets.min().item(), epoch * len(val_loader))
                writer.add_scalar("Validation/Target_Max", val_targets.max().item(), epoch * len(val_loader))

                all_preds.append(val_preds)
                all_targets.append(val_targets)

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Validation/Loss", avg_val_loss, epoch)

        preds_tensor = torch.cat(all_preds, dim=0).squeeze(1).cpu().numpy()
        targets_tensor = torch.cat(all_targets, dim=0).squeeze(1).cpu().numpy()

        val_mae = mae(preds_tensor, targets_tensor)
        val_rmse = rmse(preds_tensor, targets_tensor)
        deltas = delta_metrics(torch.tensor(preds_tensor), torch.tensor(targets_tensor))
        val_nmad = nmad(preds_tensor, targets_tensor)  # Calcola NMAD

        writer.add_scalar("Validation/MAE", val_mae, epoch)
        writer.add_scalar("Validation/RMSE", val_rmse, epoch)
        writer.add_scalar("Validation/Delta1", deltas["delta1"], epoch)
        writer.add_scalar("Validation/Delta2", deltas["delta2"], epoch)
        writer.add_scalar("Validation/Delta3", deltas["delta3"], epoch)
        writer.add_scalar("Validation/NMAD", val_nmad, epoch)  # Log NMAD

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss:"
            f" {avg_val_loss:.4f} - MAE: {val_mae:.4f} - RMSE: {val_rmse:.4f} - NMAD: {val_nmad:.4f} - δ1:"
            f" {deltas['delta1']:.3f} - δ2: {deltas['delta2']:.3f} - δ3:"
            f" {deltas['delta3']:.3f} - Time: {epoch_time:.2f}s"
        )

        # Salva le predizioni
        if (epoch + 1) % SAVE_IMG_EVERY == 0:
            save_predictions(val_preds, val_targets, epoch, writer)  # Passa l'oggetto writer

        # Logging CSV (aggiunto NMAD)
        with open(LOG_CSV_SAVE_PATH, "a") as f:
            f.write(
                f"{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_mae:.4f},{val_rmse:.4f},{val_nmad:.4f},{deltas['delta1']:.3f},{deltas['delta2']:.3f},{deltas['delta3']:.3f},{epoch_time:.2f}\n"
            )  # Include NMAD

        scheduler.step(avg_val_loss)

        # Early stopping (salvataggio migliorato del checkpoint)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_epochs = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'no_improve_epochs': no_improve_epochs
            }
            torch.save(checkpoint, BEST_MODEL_SAVE_PATH)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print("🛑 Early stopping.")
                break

        # Salva l'ultimo checkpoint (anche qui salvataggio migliorato)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'no_improve_epochs': no_improve_epochs
        }
        torch.save(checkpoint, LAST_MODEL_SAVE_PATH)

    print("Training completo ✅")
    writer.close()  # Chiude lo scrittore di TensorBoard