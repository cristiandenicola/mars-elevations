# Hyperparameters and paths
PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-5
SCHEDULER_PATIENCE = 5
EARLY_STOP_PATIENCE = 10
LR_FACTOR = 0.5
SEED = 42

DEVICE = "mps"

# Data paths
DATA_DIR = "/Users/cristiandenicola/Documents/data/mars_datasets/"
CASSIS_PAN = f"{DATA_DIR}/PAN"
CASSIS_DTM = f"{DATA_DIR}/DTM"

# Model and infos paths
BEST_MODEL_SAVE_PATH = "best_model.pth"
LAST_MODEL_SAVE_PATH = "last_model.pth"

LOG_CSV_SAVE_PATH = "training_log.csv"

PRED_SAVE_DIR = "train_pred"

