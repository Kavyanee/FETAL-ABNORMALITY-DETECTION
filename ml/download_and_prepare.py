"""
Download and prepare dataset from Kaggle
"""
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

print("="*70)
print("📥 DOWNLOADING DATASET FROM KAGGLE")
print("="*70)

# Step 1: Download from Kaggle
import kagglehub
print("\n🔄 Downloading dataset...")
kaggle_path = kagglehub.dataset_download("orvile/ultrasound-fetus-dataset")
print(f"✅ Downloaded to: {kaggle_path}")

# Step 2: Find train folder
print("\n🔍 Searching for train folder...")
train_path = None
for root, dirs, files in os.walk(kaggle_path):
    if 'train' in dirs:
        train_path = os.path.join(root, 'train')
        break
    # Check if current folder has class folders
    dir_lower = [d.lower() for d in dirs]
    if 'normal' in dir_lower or 'benign' in dir_lower or 'malignant' in dir_lower:
        train_path = root
        break

if not train_path:
    print("❌ Could not find train folder")
    print(f"Please check: {kaggle_path}")
    sys.exit(1)

print(f"✅ Found train folder: {train_path}")

# Step 3: List what's in train folder
print("\n📂 Contents of train folder:")
if os.path.exists(train_path):
    for item in os.listdir(train_path):
        item_path = os.path.join(train_path, item)
        if os.path.isdir(item_path):
            # Check all image extensions
            jpg_count = len(list(Path(item_path).glob('*.jpg')))
            jpeg_count = len(list(Path(item_path).glob('*.jpeg')))
            png_count = len(list(Path(item_path).glob('*.png')))
            total = jpg_count + jpeg_count + png_count
            print(f"   {item}: {total} images (jpg:{jpg_count}, jpeg:{jpeg_count}, png:{png_count})")
            # Show first few files
            all_files = list(Path(item_path).iterdir())[:5]
            if all_files:
                print(f"      Sample files: {[f.name for f in all_files]}")

# Step 4: Copy images
print("\n📋 Copying images to processed folder...")
print(f"Destination: {config.PROCESSED_DATA_DIR}")

# Try different case variations
class_mappings = [
    ('Normal', 'Normal'),
    ('normal', 'Normal'),
    ('Benign', 'Benign'),
    ('benign', 'Benign'),
    ('Malignant', 'Malignant'),
    ('malignant', 'Malignant')
]

copied_classes = set()
for src_name, dst_name in class_mappings:
    if dst_name in copied_classes:
        continue
        
    src_path = os.path.join(train_path, src_name)
    dst_path = os.path.join(config.PROCESSED_DATA_DIR, dst_name)
    
    if not os.path.exists(src_path):
        continue
    
    os.makedirs(dst_path, exist_ok=True)
    # Try all image extensions
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        images.extend(list(Path(src_path).glob(ext)))
    images = images[:500]
    
    if len(images) == 0:
        continue
    
    print(f"\n📸 Copying {dst_name}: {len(images)} images")
    copied_classes.add(dst_name)
    
    for img_path in tqdm(images, desc=dst_name):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, (config.IMG_HEIGHT, config.IMG_WIDTH))
                cv2.imwrite(os.path.join(dst_path, img_path.name), img)
        except Exception as e:
            print(f"Error: {e}")

# Step 5: Verify
print("\n✅ Verification:")
for class_name in config.CLASS_NAMES:
    class_dir = os.path.join(config.PROCESSED_DATA_DIR, class_name)
    if os.path.exists(class_dir):
        # Check all extensions
        count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            count += len(list(Path(class_dir).glob(ext)))
        print(f"   {class_name}: {count} images")

print("\n✅ Done! Now run: python train_federated.py")
