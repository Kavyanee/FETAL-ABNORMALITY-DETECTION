"""
Data Preprocessing Module

This module handles:
1. Dataset download from Kaggle
2. Image preprocessing (resize, normalize)
3. Data splitting (train/val/test)
4. Federated data partitioning (simulating multiple hospitals)
5. Data augmentation

WHY THIS IS NEEDED:
- Medical images need standardization (size, format)
- Federated Learning requires data to be split across clients
- Augmentation helps with limited medical data
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from tqdm import tqdm
import kagglehub

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class DataPreprocessor:
    """
    Handles all data preprocessing tasks for fetal ultrasound images
    """
    
    def __init__(self):
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        self.img_channels = IMG_CHANNELS
        
    def download_dataset(self):
        """
        Download ultrasound dataset from Kaggle
        
        WHY: We need a real ultrasound image dataset for training
        NOTE: This downloads to Kaggle's cache, we'll copy to our structure
        """
        print("[*] Downloading dataset from Kaggle...")
        try:
            # Download the dataset
            path = kagglehub.dataset_download("orvile/ultrasound-fetus-dataset")
            print(f"[*] Dataset downloaded to: {path}")
            return path
        except Exception as e:
            print(f"[*] Error downloading dataset: {e}")
            print("[*] Make sure you have Kaggle API configured:")
            print("   1. Create account on kaggle.com")
            print("   2. Go to Account > API > Create New Token")
            print("   3. Place kaggle.json in ~/.kaggle/")
            return None
    
    def organize_dataset(self, source_path):
        """
        Organize downloaded images into Normal/Benign/Malignant structure
        """
        print("\n[*] Organizing dataset into class categories...")
        
        # Create directory structure for each class
        class_dirs = {}
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            class_dirs[class_name] = class_dir
        
        # Map source folders to our class names (case-insensitive)
        folder_mapping = {
            'normal': 'Normal',
            'benign': 'Benign',
            'malignant': 'Malignant'
        }
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        class_counts = {class_name: 0 for class_name in CLASS_NAMES}
        
        # Walk through source directory
        for root, dirs, files in os.walk(source_path):
            # Check if current folder matches any class
            folder_name = os.path.basename(root).lower()
            
            if folder_name in folder_mapping:
                target_class = folder_mapping[folder_name]
                print(f"[*] Processing {target_class} images from {root}...")
                
                for file in tqdm(files):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        img_path = os.path.join(root, file)
                        output_name = f"{folder_name}_{class_counts[target_class]}.jpg"
                        
                        if self._preprocess_and_save(img_path, class_dirs[target_class], output_name):
                            class_counts[target_class] += 1
        
        total_images = sum(class_counts.values())
        
        if total_images == 0:
            print("[*] No images found. Creating synthetic dataset...")
            return self._create_synthetic_dataset()
        
        print(f"[*] Dataset organized:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} images")
        
        return class_counts
    
    def _preprocess_and_save(self, img_path, output_dir, output_name):
        """
        Preprocess a single image and save it
        
        Steps:
        1. Read image
        2. Resize to standard size
        3. Normalize pixel values
        4. Save to output directory
        """
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                return False
            
            # Resize
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # Save
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, img)
            return True
        except Exception as e:
            print(f"[*]  Error processing {img_path}: {e}")
            return False
    
    def _create_synthetic_dataset(self):
        """
        Create synthetic ultrasound-like images for demo purposes
        
        WHY: Fallback if Kaggle download fails
        NOTE: This is only for demonstration - not for real medical use
        """
        print("[*] Creating synthetic dataset...")
        
        normal_dir = os.path.join(PROCESSED_DATA_DIR, 'Normal')
        abnormal_dir = os.path.join(PROCESSED_DATA_DIR, 'Abnormal')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(abnormal_dir, exist_ok=True)
        
        # Create 200 normal and 100 abnormal synthetic images
        for i in range(200):
            img = self._generate_synthetic_ultrasound(is_normal=True)
            cv2.imwrite(os.path.join(normal_dir, f"normal_{i}.jpg"), img)
        
        for i in range(100):
            img = self._generate_synthetic_ultrasound(is_normal=False)
            cv2.imwrite(os.path.join(abnormal_dir, f"abnormal_{i}.jpg"), img)
        
        print("[*] Synthetic dataset created (200 Normal, 100 Abnormal)")
        return 200, 100
    
    def _generate_synthetic_ultrasound(self, is_normal=True):
        """
        Generate a synthetic ultrasound-like image
        """
        # Create grayscale noise base
        img = np.random.randint(50, 150, (self.img_height, self.img_width), dtype=np.uint8)
        
        # Add some structure (ellipse for fetus)
        center = (self.img_width // 2, self.img_height // 2)
        axes = (60, 80) if is_normal else (40, 100)  # Abnormal has different proportions
        cv2.ellipse(img, center, axes, 0, 0, 360, 180, -1)
        
        # Add noise
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
    
    def create_federated_splits(self, num_clients=NUM_CLIENTS):
        """
        Split dataset across multiple clients (hospitals) for Federated Learning
        
        WHY: Each hospital has its own local dataset
        HOW: We use Dirichlet distribution for non-IID split (realistic)
        
        Non-IID means: Each hospital has different data distributions
        (e.g., Hospital A has more abnormal cases than Hospital B)
        """
        print(f"\n[*] Creating federated data splits for {num_clients} hospitals...")
        
        # Load all image paths and labels from all class directories
        all_images = []
        all_labels = []
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
            if os.path.exists(class_dir):
                class_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
                all_images.extend(class_images)
                all_labels.extend([class_idx] * len(class_images))
        
        # Shuffle
        indices = np.random.permutation(len(all_images))
        all_images = [all_images[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        
        # Split into clients (simple equal split for demo)
        images_per_client = len(all_images) // num_clients
        
        for client_id in range(num_clients):
            start_idx = client_id * images_per_client
            end_idx = start_idx + images_per_client if client_id < num_clients - 1 else len(all_images)
            
            client_images = all_images[start_idx:end_idx]
            client_labels = all_labels[start_idx:end_idx]
            
            # Create client directory
            client_dir = os.path.join(FEDERATED_DATA_DIR, f'hospital_{client_id}')
            os.makedirs(client_dir, exist_ok=True)
            
            # Save client data info
            client_data = pd.DataFrame({
                'image_path': client_images,
                'label': client_labels
            })
            client_data.to_csv(os.path.join(client_dir, 'data.csv'), index=False)
            
            # Count per class
            class_counts = {class_name: sum(1 for l in client_labels if l == idx) 
                          for idx, class_name in enumerate(CLASS_NAMES)}
            counts_str = ', '.join([f"{name}: {count}" for name, count in class_counts.items()])
            
            print(f"   Hospital {client_id}: {len(client_images)} images ({counts_str})")
        
        print("[*] Federated splits created")
    
    def load_client_data(self, client_id):
        """
        Load data for a specific client (hospital) - MEMORY EFFICIENT VERSION
        
        Uses tf.data.Dataset to avoid loading entire dataset into memory at once.
        This prevents OutOfMemory errors on GPU/CPU during federated rounds.
        
        Returns: (train_dataset, val_dataset, X_val_small, y_val_small)
        """
        from tensorflow.keras.utils import to_categorical
        import tensorflow as tf
        
        client_dir = os.path.join(FEDERATED_DATA_DIR, f'hospital_{client_id}')
        data_csv = os.path.join(client_dir, 'data.csv')
        
        df = pd.read_csv(data_csv)
        
        # Convert labels to one-hot encoding
        y = df['label'].values
        
        # Split into train/val INDICES
        train_idx, val_idx = train_test_split(
            range(len(df)), 
            test_size=0.2, 
            random_state=RANDOM_SEED, 
            stratify=y
        )
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        
        y_train = to_categorical(df_train['label'].values, num_classes=NUM_CLASSES)
        y_val = to_categorical(df_val['label'].values, num_classes=NUM_CLASSES)
        
        # Create tf.data datasets
        train_dataset = self._create_tf_dataset(
            df_train, y_train, augment=True, batch_size=LOCAL_BATCH_SIZE
        )
        
        val_dataset = self._create_tf_dataset(
            df_val, y_val, augment=False, batch_size=LOCAL_BATCH_SIZE
        )
        
        # Also load small validation set for evaluation
        X_val_small = []
        y_val_small = []
        for idx, row in df_val.iterrows():
            img = cv2.imread(row['image_path'])
            if img is not None:
                img = cv2.resize(img, (self.img_width, self.img_height))
                img = img / 255.0
                X_val_small.append(img)
                y_val_small.append(row['label'])
        
        X_val_small = np.array(X_val_small)
        y_val_small = to_categorical(np.array(y_val_small), num_classes=NUM_CLASSES)
        
        return train_dataset, val_dataset, X_val_small, y_val_small
    
    def _create_tf_dataset(self, df, y, augment=True, batch_size=32):
        """
        Create a tf.data.Dataset that loads images on-the-fly
        
        This prevents loading entire dataset into memory
        """
        import tensorflow as tf
        
        def load_and_preprocess_image(path, label):
            """Load image from path and apply preprocessing"""
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (self.img_height, self.img_width))
            img = img / 255.0  # Normalize to [0, 1]
            
            if augment:
                # Random augmentation
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)
                img = tf.image.random_brightness(img, 0.2)
                img = tf.image.random_contrast(img, 0.8, 1.2)
            
            return img, label
        
        # Create dataset from tensors
        image_paths = tf.constant(df['image_path'].values)
        labels = tf.constant(y, dtype=tf.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.shuffle(buffer_size=min(1000, len(df)))
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_data_augmentation(self):
        """
        Create data augmentation generator
        
        WHY: Medical datasets are often small, augmentation helps
        WHAT: Random rotations, shifts, flips to create variations
        """
        return ImageDataGenerator(**AUGMENTATION_CONFIG)


def main():
    """
    Main preprocessing pipeline
    """
    print("=" * 70)
    print("[*] FETAL ABNORMALITY DETECTION - DATA PREPROCESSING")
    print("=" * 70)
    
    preprocessor = DataPreprocessor()
    
    # Step 1: Download dataset
    dataset_path = preprocessor.download_dataset()
    
    # Step 2: Organize into Normal/Abnormal
    if dataset_path:
        preprocessor.organize_dataset(dataset_path)
    else:
        # Fallback to synthetic data
        preprocessor._create_synthetic_dataset()
    
    # Step 3: Create federated splits
    preprocessor.create_federated_splits()
    
    print("\n[*] Preprocessing complete!")
    print(f"[*] Processed data saved to: {PROCESSED_DATA_DIR}")
    print(f"[*] Federated data saved to: {FEDERATED_DATA_DIR}")


if __name__ == "__main__":
    main()
