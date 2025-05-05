import matplotlib.pyplot as plt
import numpy as np


def show_prediction(input_img, pred, gt):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(input_img, cmap="gray")
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(pred, cmap="terrain")
    axs[1].set_title("Predicted Elevation")
    axs[1].axis("off")

    axs[2].imshow(gt, cmap="terrain")
    axs[2].set_title("Ground Truth Elevation")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
