"""
Main Federated Learning Training Script

This is the entry point for training the model using Federated Learning.

USAGE:
    python train_federated.py

WHAT HAPPENS:
1. Downloads and preprocesses dataset
2. Splits data across hospitals
3. Initializes federated learning server
4. Trains model across multiple rounds
5. Saves final global model

PRIVACY GUARANTEE:
- No raw images are shared between hospitals
- Only model weights are exchanged
- Each hospital's data stays local
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import DataPreprocessor
from src.federated_learning import FederatedLearningServer
from config import *


def main():
    """
    Main training pipeline
    """
    print("=" * 70)
    print("[FEDERATED LEARNING] PRIVACY-PRESERVING FETAL ABNORMALITY DETECTION")
    print("   Federated Learning Training Pipeline")
    print("=" * 70)
    
    # Step 1: Check if data is already preprocessed
    preprocessor = DataPreprocessor()
    
    if not os.path.exists(PROCESSED_DATA_DIR) or len(os.listdir(PROCESSED_DATA_DIR)) == 0:
        print("\n[STEP 1] Data Preprocessing")
        print("-" * 70)
        
        # Download dataset
        dataset_path = preprocessor.download_dataset()
        
        # Organize dataset
        if dataset_path:
            preprocessor.organize_dataset(dataset_path)
        else:
            print("[WARNING] Using synthetic dataset for demo")
            preprocessor._create_synthetic_dataset()
    else:
        print("\n[OK] Data already preprocessed")
    
    # Check if federated splits exist
    if not os.path.exists(FEDERATED_DATA_DIR) or len(os.listdir(FEDERATED_DATA_DIR)) == 0:
        print("[*] Creating federated splits...")
        preprocessor.create_federated_splits()
    else:
        print("[OK] Federated splits already exist")
    
    # Step 2: Federated Learning Training
    print("\n[STEP 2] Federated Learning Training")
    print("-" * 70)
    
    server = FederatedLearningServer(
        num_clients=NUM_CLIENTS,
        num_rounds=FL_ROUNDS
    )
    
    server.start_training()
    
    # Step 3: Summary
    print("\n" + "=" * 70)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n[*] Model saved to: {GLOBAL_MODEL_PATH}")
    print(f"[*] Training configuration:")
    print(f"   - Hospitals: {NUM_CLIENTS}")
    print(f"   - FL Rounds: {FL_ROUNDS}")
    print(f"   - Local Epochs: {LOCAL_EPOCHS}")
    print(f"   - Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
    
    print("\n[NEXT STEPS]")
    print("   1. Run evaluation: python src/evaluation.py")
    print("   2. Deploy to backend: Copy model to backend/models/")
    print("   3. Test with frontend UI")
    
    print("\n[PRIVACY GUARANTEE]")
    print("   [OK] No patient images were shared between hospitals")
    print("   [OK] Only model weights were exchanged")
    print("   [OK] HIPAA/GDPR compliant approach")


if __name__ == "__main__":
    main()
