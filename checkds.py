import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.mars_dataset import *
from config import *

# Crea un'istanza del tuo dataset
real_mars_dataset = RealMarsDataset(CASSIS_PAN, CASSIS_DTM)

# Crea un DataLoader per iterare sul dataset (batch_size=1 per visualizzare singole coppie)
dataloader = DataLoader(real_mars_dataset, batch_size=1, shuffle=False)

# Numero di coppie da visualizzare
num_samples_to_plot = 50  # Puoi cambiare questo numero

# Visualizza alcune coppie PAN-DTM
for i, (pan_tensor, dtm_tensor, filename) in enumerate(dataloader):
    if i >= num_samples_to_plot:
        break

    pan = pan_tensor.squeeze().numpy()
    dtm = dtm_tensor.squeeze().numpy()

    print(f"Visualizzando immagine: {filename[0]}")

    # Crea la figura e gli assi per il plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Sample: {filename[0]}", fontsize=14)

    # Plotta l'immagine PAN
    im1 = axes[0].imshow(pan, cmap='gray')
    axes[0].set_title("PAN")
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], shrink=0.8)

    # Plotta il DTM
    im2 = axes[1].imshow(dtm, cmap='viridis')
    axes[1].set_title("DTM")
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.show()

print("✅ Visualizzazione completata.")