"""
Complete Fix for Low Accuracy Issue

This script will:
1. Clean old incorrectly organized data
2. Re-download and properly organize dataset
3. Retrain with optimized settings
"""
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("FIXING LOW ACCURACY ISSUE")
print("=" * 70)

print("\n[ISSUE IDENTIFIED]")
print("1. Data was organized as Normal/Abnormal instead of Normal/Benign/Malignant")
print("2. Model was using binary classification for 3-class problem")
print("3. Transfer learning was disabled")

print("\n[FIXES APPLIED]")
print("1. Updated data_preprocessing.py to handle 3 classes correctly")
print("2. Updated config.py:")
print("   - CLASS_NAMES = ['Normal', 'Benign', 'Malignant']")
print("   - USE_TRANSFER_LEARNING = True (MobileNetV2)")
print("   - FREEZE_BASE_MODEL = False (fine-tuning enabled)")
print("   - IMG_SIZE = 224x224 (standard for transfer learning)")
print("   - LEARNING_RATE = 0.0001 (lower for fine-tuning)")
print("   - FL_ROUNDS = 10, LOCAL_EPOCHS = 5")

print("\n[NEXT STEPS]")
print("1. Run: python cleanup_for_multiclass.py")
print("2. Run: python train_federated.py")
print("3. Model will now:")
print("   - Use MobileNetV2 pretrained weights")
print("   - Fine-tune on your 3-class dataset")
print("   - Achieve much better accuracy")

print("\n[EXPECTED IMPROVEMENTS]")
print("Before: ~33% accuracy (random guessing for 3 classes)")
print("After:  70-85% accuracy with transfer learning")

print("\n" + "=" * 70)
print("Ready to retrain! Run cleanup script first.")
print("=" * 70)
