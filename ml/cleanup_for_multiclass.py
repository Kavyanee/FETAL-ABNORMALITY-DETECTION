"""
Script to clean old binary classification data and prepare for 3-class classification
"""
import os
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
FEDERATED_DIR = os.path.join(DATA_DIR, 'federated')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

print("=" * 70)
print("CLEANING OLD DATA FOR 3-CLASS CLASSIFICATION")
print("=" * 70)

# Remove old processed data
if os.path.exists(PROCESSED_DIR):
    print(f"\n[*] Removing old processed data: {PROCESSED_DIR}")
    shutil.rmtree(PROCESSED_DIR)
    print("[OK] Removed")

# Remove old federated splits
if os.path.exists(FEDERATED_DIR):
    print(f"[*] Removing old federated splits: {FEDERATED_DIR}")
    shutil.rmtree(FEDERATED_DIR)
    print("[OK] Removed")

# Remove old models
if os.path.exists(MODELS_DIR):
    for file in os.listdir(MODELS_DIR):
        if file.endswith('.h5'):
            filepath = os.path.join(MODELS_DIR, file)
            print(f"[*] Removing old model: {filepath}")
            os.remove(filepath)
            print("[OK] Removed")

# Recreate directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FEDERATED_DIR, exist_ok=True)

print("\n" + "=" * 70)
print("[SUCCESS] Cleanup complete!")
print("=" * 70)
print("\nNext steps:")
print("1. Run: python train_federated.py")
print("2. The script will download and organize data into 3 classes:")
print("   - Normal")
print("   - Benign")
print("   - Malignant")
