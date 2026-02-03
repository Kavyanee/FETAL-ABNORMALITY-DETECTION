"""
Centralized Training Script (Baseline)

This script trains the model using traditional centralized approach
for comparison with federated learning.

WHY THIS EXISTS:
- Baseline to compare federated learning performance
- Shows the privacy trade-off

DIFFERENCE FROM FEDERATED:
- All data is pooled together (privacy violation in real scenario)
- Single training process
- Typically faster convergence
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from src.data_preprocessing import DataPreprocessor
from src.cnn_model import create_model
from config import *


def load_all_data():
    """
    Load and combine data from all hospitals (centralized approach)
    
    NOTE: This violates privacy - only for comparison!
    """
    import os
    import cv2
    import pandas as pd
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    
    print("📥 Loading data from all hospitals (centralized)...")
    
    preprocessor = DataPreprocessor()
    
    X_train_all = []
    y_train_all = []
    X_val_all = []
    y_val_all = []
    
    # Combine data from all clients
    for client_id in range(NUM_CLIENTS):
        print(f"   Loading Hospital {client_id}...")
        
        # Load client CSV
        client_dir = os.path.join(FEDERATED_DATA_DIR, f'hospital_{client_id}')
        data_csv = os.path.join(client_dir, 'data.csv')
        
        df = pd.read_csv(data_csv)
        
        # Split into train/val
        y = df['label'].values
        train_idx, val_idx = train_test_split(
            range(len(df)), 
            test_size=0.2, 
            random_state=RANDOM_SEED, 
            stratify=y
        )
        
        # Load training images
        X_train_client = []
        y_train_client = []
        for idx in train_idx:
            img_path = df.iloc[idx]['image_path']
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    img = img / 255.0
                    X_train_client.append(img)
                    y_train_client.append(df.iloc[idx]['label'])
        
        # Load validation images
        X_val_client = []
        y_val_client = []
        for idx in val_idx:
            img_path = df.iloc[idx]['image_path']
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    img = img / 255.0
                    X_val_client.append(img)
                    y_val_client.append(df.iloc[idx]['label'])
        
        if len(X_train_client) > 0:
            X_train_all.append(np.array(X_train_client))
            y_train_all.append(to_categorical(np.array(y_train_client), num_classes=NUM_CLASSES))
        
        if len(X_val_client) > 0:
            X_val_all.append(np.array(X_val_client))
            y_val_all.append(to_categorical(np.array(y_val_client), num_classes=NUM_CLASSES))
    
    # Concatenate
    if len(X_train_all) > 0:
        X_train = np.concatenate(X_train_all, axis=0)
        y_train = np.concatenate(y_train_all, axis=0)
    else:
        raise ValueError("No training data loaded!")
    
    if len(X_val_all) > 0:
        X_val = np.concatenate(X_val_all, axis=0)
        y_val = np.concatenate(y_val_all, axis=0)
    else:
        raise ValueError("No validation data loaded!")
    
    print(f"✅ Combined dataset:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    return X_train, y_train, X_val, y_val


def main():
    """
    Centralized training pipeline
    """
    # Set encoding for Windows emoji support
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 70)
    print("[CENTRALIZED TRAINING - BASELINE]")
    print("   WARNING: All data pooled together")
    print("=" * 70)
    
    # Load all data
    X_train, y_train, X_val, y_val = load_all_data()
    
    # Create model
    print("\n[*] Creating model...")
    model = create_model()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            verbose=1
        )
    ]
    
    # Compute class weights to handle imbalance
    class_weights = None
    if USE_CLASS_WEIGHTS and AUTOMATIC_CLASS_WEIGHTS:
        print("\n[*] Computing class weights (to handle imbalance)...")
        y_train_labels = np.argmax(y_train, axis=1)
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        class_weights = {i: w for i, w in enumerate(class_weights_array)}
        print(f"   Class weights: {class_weights}")
        print(f"      Normal: {class_weights[0]:.2f}x (minority)")
        print(f"      Benign: {class_weights[1]:.2f}x (minority)")
        print(f"      Malignant: {class_weights[2]:.2f}x (majority)")
    
    # Train
    print("\n[*] Starting training...")
    print(f"   Epochs: {EPOCHS_CENTRALIZED}")
    print(f"   Batch size: {BATCH_SIZE}")
    print("-" * 70)
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_CENTRALIZED,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save model
    model.save(CENTRALIZED_MODEL_PATH)
    
    print("\n" + "=" * 70)
    print("[OK] CENTRALIZED TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n[*] Model saved to: {CENTRALIZED_MODEL_PATH}")
    print(f"\n[*] Final Metrics:")
    print(f"   Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"   Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    print("\n[*] NOTE:")
    print("   This approach requires sharing all patient data to a central server")
    print("   NOT suitable for real medical applications due to privacy concerns")


if __name__ == "__main__":
    main()
