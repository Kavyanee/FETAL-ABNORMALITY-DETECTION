"""
Configuration file for the Fetal Abnormality Detection System

This file contains all hyperparameters, paths, and settings.
Modify these values to experiment with different configurations.
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
FEDERATED_DATA_DIR = os.path.join(DATA_DIR, 'federated')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  FEDERATED_DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Image preprocessing
IMG_HEIGHT = 224  # Standard size for transfer learning
IMG_WIDTH = 224
IMG_CHANNELS = 3  # RGB images
BATCH_SIZE = 16

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Class labels
CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================================
# FEDERATED LEARNING CONFIGURATION
# ============================================================================
# Number of simulated hospitals (clients)
NUM_CLIENTS = 3

# Federated learning rounds
FL_ROUNDS = 10  # Increased for better convergence

# Local training configuration per client
LOCAL_EPOCHS = 5  # Increased back (we need more epochs to learn properly)
LOCAL_BATCH_SIZE = 16  # Increased from 8 (more stable gradients)

# Minimum samples per client (for realistic simulation)
MIN_SAMPLES_PER_CLIENT = 50

# Federated averaging strategy
AGGREGATION_STRATEGY = 'FedAvg'  # Federated Averaging

# ============================================================================
# CNN MODEL CONFIGURATION
# ============================================================================
# Model architecture choice
USE_TRANSFER_LEARNING = False  # DISABLED: MobileNetV2 doesn't work for ultrasound images
# Custom CNN works better for medical images

# Custom CNN architecture (if USE_TRANSFER_LEARNING = False)
CONV_FILTERS = [32, 64, 128, 128]  # Added extra conv layer for more feature extraction
DENSE_UNITS = [256, 128]  # Increased for complex decision making
DROPOUT_RATE = 0.3  # Reduced (0.4 was too much, preventing learning)
L2_REGULARIZATION = 0.00001  # Very light (0.00005 was too strong)

# Transfer learning model (if USE_TRANSFER_LEARNING = True)
TRANSFER_MODEL = 'MobileNetV2'
FREEZE_BASE_MODEL = False  # Fine-tune for better accuracy

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Optimizer
LEARNING_RATE = 0.0005  # Sweet spot: faster than 0.0001, stable unlike 0.001
OPTIMIZER = 'adam'

# Loss function
LOSS_FUNCTION = 'categorical_crossentropy'  # For multi-class classification

# Metrics
METRICS = ['accuracy', 'precision', 'recall']

# Training parameters
EPOCHS_CENTRALIZED = 50  # Restored: Need more epochs for custom CNN to learn
EARLY_STOPPING_PATIENCE = 15  # Give model more time to learn
EARLY_STOPPING_ENABLED = False  # DISABLED: Was stopping too early with poor accuracy
REDUCE_LR_PATIENCE = 5

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
# Augmentation parameters (to increase dataset diversity)
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'fill_mode': 'nearest'
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# Metrics to compute
EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']

# Confusion matrix
SAVE_CONFUSION_MATRIX = True

# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================
# Handle class imbalance: Malignant (4507) vs Normal (750) vs Benign (776)
USE_CLASS_WEIGHTS = True  # ENABLED: Helps balance 3-class prediction
AUTOMATIC_CLASS_WEIGHTS = True


# ============================================================================
# PRIVACY CONFIGURATION
# ============================================================================
# Differential privacy (future enhancement)
USE_DIFFERENTIAL_PRIVACY = False
DP_EPSILON = 1.0  # Privacy budget
DP_DELTA = 1e-5

# Secure aggregation
USE_SECURE_AGGREGATION = False  # Requires advanced Flower setup

# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================
VERBOSE = 1  # 0: silent, 1: progress bar, 2: one line per epoch
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# MODEL EXPORT
# ============================================================================
GLOBAL_MODEL_PATH = os.path.join(MODELS_DIR, 'global_model.h5')
CENTRALIZED_MODEL_PATH = os.path.join(MODELS_DIR, 'centralized_model.h5')

# Export format for backend
EXPORT_FORMAT = 'h5'  # Options: 'h5', 'savedmodel'

print("[OK] Configuration loaded successfully")
print(f"[*] Models will be saved to: {MODELS_DIR}")
print(f"[*] Dataset directory: {DATA_DIR}")
print(f"[*] Number of federated clients: {NUM_CLIENTS}")
print(f"[*] Federated learning rounds: {FL_ROUNDS}")
