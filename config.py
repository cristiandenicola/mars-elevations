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

# Dataset parameters
TARGET_DTM_NORMALIZATION_RANGE = 600.0
GLOBAL_DTM_MAX_ABS = 600.0
DTM_PREDICTION_MARGIN = 50.0
GLOBAL_PAN_STD = 13933.474609
GLOBAL_PAN_MEAN = 156.814468


