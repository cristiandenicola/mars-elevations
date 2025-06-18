# --- Device Configuration ---
DEVICE = "mps"

# --- Dataset Paths ---
DATA_DIR = "/Users/cristiandenicola/Documents/data/mars_datasets/"
CASSIS_PAN = f"{DATA_DIR}/PAN"
CASSIS_DTM = f"{DATA_DIR}/DTM"

# --- DTM Normalization Parameters ---
# DTMs are normalized to this range. E.g., a DTM value of X will be mapped to [0, TARGET_DTM_NORMALIZATION_RANGE]
TARGET_DTM_NORMALIZATION_RANGE = 600.0 
# Non useremo più GLOBAL_DTM_MAX_ABS e DTM_PREDICTION_MARGIN nel modello stesso
# ma li manteniamo qui se li usi nel tuo script di pre-processing o altrove.
GLOBAL_DTM_MAX_ABS = 600.0 
DTM_PREDICTION_MARGIN = 0.0
GLOBAL_PAN_STD = 13933.474609
GLOBAL_PAN_MEAN = 156.814468

# --- Model Parameters ---
# Swin Transformer V2 Base, pre-addestrato su immagini 256x256
# timm_docs: https://rwightman.github.io/pytorch-image-models/models/swin/
ENCODER_NAME = "swinv2_small_window16_256" 
PRETRAINED = True 

# --- Training Parameters ---
PATCH_SIZE = 256
STRIDE = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
SCHEDULER_PATIENCE = 5
EARLY_STOP_PATIENCE = 15
LR_FACTOR = 0.5
WEIGHT_DECAY = 1e-2
SEED = 42

# --- Paths ---
BEST_MODEL_SAVE_PATH = "best_model.pth"
LAST_MODEL_SAVE_PATH = "last_model.pth"
LOG_CSV_SAVE_PATH = "training_log.csv"
PRED_SAVE_DIR = "train_pred"