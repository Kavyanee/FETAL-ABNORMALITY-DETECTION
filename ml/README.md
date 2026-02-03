# ML Pipeline - Privacy-Preserving Fetal Abnormality Detection

## Overview
This folder contains the complete machine learning pipeline including:
- Data preprocessing
- CNN model architecture
- Federated Learning implementation
- Model evaluation

## Structure
```
ml/
├── data/                    # Dataset storage (gitignored)
│   ├── raw/                # Original images
│   ├── processed/          # Preprocessed images
│   └── federated/          # Hospital-wise splits
├── models/                  # Saved models
│   ├── global_model.h5     # Final federated model
│   └── checkpoints/        # Training checkpoints
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── cnn_model.py
│   ├── federated_learning.py
│   └── evaluation.py
├── config.py               # Configuration parameters
├── train_federated.py      # Main FL training script
├── train_centralized.py    # Baseline centralized training
└── requirements.txt        # Python dependencies
```

## Privacy Guarantees
- Raw images never leave hospital premises
- Only model weights are shared
- Differential privacy can be added (future work)
- Secure aggregation protocol

## Usage
See individual scripts for detailed usage instructions.
