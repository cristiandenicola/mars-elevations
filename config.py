import torch
import os

# Training hyperparameters
PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-5
SCHEDULER_PATIENCE = 5
EARLY_STOP_PATIENCE = 10
LR_FACTOR = 0.5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
CASSIS_PAN = "/content/dataset_pan_patches/PAN"
CASSIS_DTM = "/content/dataset_dtm_patches/DTM"

COLAB_SAVE_ROOT = "/content/drive/MyDrive/MARS"

BEST_MODEL_SAVE_PATH = os.path.join(COLAB_SAVE_ROOT, "best_model.pth")
LAST_MODEL_SAVE_PATH = os.path.join(COLAB_SAVE_ROOT, "last_model.pth")
LOG_CSV_SAVE_PATH = os.path.join(COLAB_SAVE_ROOT, "training_log.csv")
PRED_SAVE_DIR = os.path.join(COLAB_SAVE_ROOT, "train_pred")

# Dataset parameters
TARGET_DTM_NORMALIZATION_RANGE = 600.0
GLOBAL_DTM_MAX_ABS = 600.0
DTM_PREDICTION_MARGIN = 50.0
GLOBAL_PAN_STD = 13933.474609
GLOBAL_PAN_MEAN = 156.814468