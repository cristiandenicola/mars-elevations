# Hyperparameters and paths
PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.0005
SEED = 42

DEVICE = "mps"

DATA_DIR = "/Users/cristiandenicola/Documents/data/r2d2/datasets/"
CASSIS_PAN = f"{DATA_DIR}/CaSSIS_dtm_pan/PAN"
CASSIS_DTM = f"{DATA_DIR}/CaSSIS_dtm_pan/DTM"

BEST_MODEL_SAVE_PATH = "best_model.pth"
LAST_MODEL_SAVE_PATH = "last_model.pth"
LOSS_CURVE_SAVE_PATH = "loss_curve.npy"
LOG_CSV_SAVE_PATH = "training_log.csv"

