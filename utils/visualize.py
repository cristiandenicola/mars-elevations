import matplotlib.pyplot as plt
import numpy as np


def show_prediction(input_img, pred, gt):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(input_img, cmap="gray")
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    im_pred = axs[1].imshow(pred, cmap="terrain")
    axs[1].set_title("Predicted Elevation")
    axs[1].axis("off")
    fig.colorbar(im_pred, ax=axs[1], label="Elevation")

    error = pred - gt
    im_error = axs[2].imshow(error, cmap="coolwarm", vmin=-np.max(np.abs(error)), vmax=np.max(np.abs(error)))
    axs[2].set_title("Prediction Error")
    axs[2].axis("off")
    fig.colorbar(im_error, ax=axs[2], label="Error")

    plt.tight_layout()
    plt.show()
