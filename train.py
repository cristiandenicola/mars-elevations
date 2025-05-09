import os
import time
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.unet import EfficientUNet
from dataset.mars_dataset import RealMarsDataset
from loss.loss import combined_loss
from evaluation.metrics import rmse, mae, nmad
from config import *
import matplotlib.pyplot as plt

# casual seed
torch.manual_seed(SEED)
np.random.seed(SEED)


dataset = RealMarsDataset(CASSIS_PAN, CASSIS_DTM)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# creo istanza del modello
# inizializzo optimizer (aggiorna pesi del modello durante il train in base al grad della loss func)
model = EfficientUNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# init scheduler
# usato x ridurre il learning rate quando la loss di train smette di migliorare
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# var train track e l'early stopping
best_loss = float("inf")
losses = []
no_improve_epochs = 0
EARLY_STOP_PATIENCE = 10

# check file di checkpoint del modello precedente (resume train)
if os.path.exists(LAST_MODEL_SAVE_PATH):
    print("⚠️  Resuming from last checkpoint...")
    model.load_state_dict(torch.load(LAST_MODEL_SAVE_PATH))

# train loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    # itero sui batch di dati forniti dal DataLoader
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        preds = model(images)
        loss = combined_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()

        # aggiorno i pesi del modello utilizzando l'ottimizzatore e i gradienti calcolati
        optimizer.step() 

        epoch_loss += loss.item()

    epoch_time = time.time() - start_time
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

    # save log csv
    with open(LOG_CSV_SAVE_PATH, "a") as f:
        f.write(f"{epoch+1},{avg_loss:.4f},{epoch_time:.2f}\n")

    # update scheduler
    scheduler.step(avg_loss)

    # save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print("🛑 Early stopping.")
            break

    # save last model
    torch.save(model.state_dict(), LAST_MODEL_SAVE_PATH)

# save curve
np.save(LOSS_CURVE_SAVE_PATH, np.array(losses))
print("Training completo ✅")
