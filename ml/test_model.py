"""
Test model predictions directly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from config import *

# Load model
print("Loading model...")
model = tf.keras.models.load_model(GLOBAL_MODEL_PATH)
print(f"Model loaded from: {GLOBAL_MODEL_PATH}")
print(f"Model output shape: {model.output_shape}")

# Test with images from each class
for class_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        print(f"⚠️  {class_name} directory not found")
        continue
    
    # Get first image
    images = list(Path(class_dir).glob('*.jpg'))[:3]
    
    print(f"\n{'='*60}")
    print(f"Testing {class_name} images:")
    print(f"{'='*60}")
    
    for img_path in images:
        # Load and preprocess
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        pred = model.predict(img, verbose=0)[0]
        pred_class = np.argmax(pred)
        
        print(f"\nImage: {img_path.name}")
        print(f"True class: {class_name} (index {class_idx})")
        print(f"Predicted: {CLASS_NAMES[pred_class]} (index {pred_class})")
        print(f"Probabilities: {pred}")
        print(f"  Normal: {pred[0]:.4f}")
        print(f"  Benign: {pred[1]:.4f}")
        print(f"  Malignant: {pred[2]:.4f}")
        
        if pred_class == class_idx:
            print("✅ CORRECT")
        else:
            print("❌ WRONG")

print(f"\n{'='*60}")
print("Test complete")
