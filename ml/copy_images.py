"""
Simple script to copy images from Kaggle dataset
"""
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

# Kaggle dataset location
kaggle_train = r"C:\Users\LENOVO\Downloads\archive\Ultrasound Fetus Dataset\Ultrasound Fetus Dataset\Data\Data\train"

print("Copying images from Kaggle dataset...")
print(f"Source: {kaggle_train}")
print(f"Destination: {config.PROCESSED_DATA_DIR}")

for src_folder, dst_folder in [('Normal', 'Normal'), ('Benign', 'Benign'), ('Malignant', 'Malignant')]:
    src_path = os.path.join(kaggle_train, src_folder)
    dst_path = os.path.join(config.PROCESSED_DATA_DIR, dst_folder)
    
    os.makedirs(dst_path, exist_ok=True)
    
    if not os.path.exists(src_path):
        print(f"❌ {src_path} not found!")
        continue
    
    images = list(Path(src_path).glob('*.jpg'))[:500]
    print(f"\nCopying {dst_folder}: {len(images)} images")
    
    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.resize(img, (config.IMG_HEIGHT, config.IMG_WIDTH))
            cv2.imwrite(os.path.join(dst_path, img_path.name), img)

print("\n✅ Done! Now run: python train_federated.py")
