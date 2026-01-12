import torch

# Training hyperparameters
PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 12
EPOCHS = 50
LEARNING_RATE = 1e-5
SCHEDULER_PATIENCE = 3
EARLY_STOP_PATIENCE = 8
LR_FACTOR = 0.7
WEIGHT_DECAY = 1e-2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths (SSH)
#CASSIS_PAN = "/home/cdenicola/my_datasets/PAN"
#CASSIS_DTM = "/home/cdenicola/my_datasets/DTM"

# Data paths
DATA_DIR = "/Users/cristiandenicola/Documents/data/mars_datasets_v2/"
CASSIS_PAN = f"{DATA_DIR}/PAN"
CASSIS_DTM = f"{DATA_DIR}/DTM"

# Model and infos paths
BEST_MODEL_SAVE_PATH = "best_model.pth"
LAST_MODEL_SAVE_PATH = "last_model.pth"
LOG_CSV_SAVE_PATH = "training_log.csv"
PRED_SAVE_DIR = "train_pred"

# Dataset parameters
TARGET_DTM_NORMALIZATION_RANGE = 600.0
GLOBAL_DTM_MAX_ABS = 600.0
DTM_PREDICTION_MARGIN = 50.0
GLOBAL_PAN_MEAN = 521.154236
GLOBAL_PAN_STD = 13494.581055

GLOBAL_DTM_MEAN = 1101.51036495
GLOBAL_DTM_STD = 4714.65057671

