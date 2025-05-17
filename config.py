# Hyperparameters and paths
PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 2e-5
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

