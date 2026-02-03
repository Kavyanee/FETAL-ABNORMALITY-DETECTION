"""
CNN Model Architecture

This module defines the Convolutional Neural Network for fetal abnormality detection.

WHY CNN FOR ULTRASOUND IMAGES?
1. Spatial Feature Learning: CNNs automatically learn hierarchical features
   - Low-level: edges, textures
   - Mid-level: shapes, patterns
   - High-level: anatomical structures

2. Translation Invariance: Abnormalities can appear anywhere in the image

3. Parameter Efficiency: Shared weights reduce overfitting on small medical datasets

4. Proven Success: CNNs are state-of-the-art for medical image analysis

ARCHITECTURE EXPLANATION:
- Convolutional Layers: Extract spatial features
- Pooling Layers: Reduce dimensionality, add robustness
- Dropout: Prevent overfitting
- Dense Layers: Classification decision
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class FetalAbnormalityCNN:
    """
    CNN Model for Binary Classification: Normal vs Abnormal
    """
    
    def __init__(self, use_transfer_learning=USE_TRANSFER_LEARNING):
        self.use_transfer_learning = use_transfer_learning
        self.model = None
        
    def build_custom_cnn(self):
        """
        Build a custom CNN from scratch
        
        ARCHITECTURE:
        Input (224x224x3)
            ↓
        Conv2D(32) → ReLU → MaxPool → Dropout
            ↓
        Conv2D(64) → ReLU → MaxPool → Dropout
            ↓
        Conv2D(128) → ReLU → MaxPool → Dropout
            ↓
        Flatten
            ↓
        Dense(256) → ReLU → Dropout
            ↓
        Dense(128) → ReLU → Dropout
            ↓
        Dense(NUM_CLASSES) → Softmax (multi-class probabilities)
        
        REGULARIZATION IMPROVEMENTS:
        - L2 regularization: Penalizes large weights to prevent overfitting
        - Higher Dropout (0.5): Randomly disables neurons during training
        - Reduced epochs: Prevent training for too long
        - Early stopping: Stop when validation loss plateaus
        - Class weights: Handle Malignant class imbalance
        """
        print("[*]  Building custom CNN architecture with regularization...")
        
        model = models.Sequential(name='FetalAbnormalityCNN')
        
        # Input layer
        model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
        
        # Convolutional Block 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                               kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION), 
                               name='conv1'))
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))
        model.add(layers.Dropout(DROPOUT_RATE))
        
        # Convolutional Block 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                               kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION), 
                               name='conv2'))
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))
        model.add(layers.Dropout(DROPOUT_RATE))
        
        # Convolutional Block 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                               kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION), 
                               name='conv3'))
        model.add(layers.MaxPooling2D((2, 2), name='pool3'))
        model.add(layers.Dropout(DROPOUT_RATE))
        
        # Convolutional Block 4 (NEW: Extra layer for more feature extraction)
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                               kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION), 
                               name='conv4'))
        model.add(layers.MaxPooling2D((2, 2), name='pool4'))
        model.add(layers.Dropout(DROPOUT_RATE))
        
        # Flatten and Dense Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu', 
                              kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION), 
                              name='dense1'))
        model.add(layers.Dropout(DROPOUT_RATE))
        model.add(layers.Dense(128, activation='relu', 
                              kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION), 
                              name='dense2'))
        model.add(layers.Dropout(DROPOUT_RATE))
        
        # Output layer (multi-class classification)
        if NUM_CLASSES == 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
        else:
            model.add(layers.Dense(NUM_CLASSES, activation='softmax', name='output'))
        
        self.model = model
        print("[*] Custom CNN built successfully")
        return model
    
    def build_transfer_learning_model(self):
        """
        Build model using Transfer Learning (MobileNetV2)
        
        WHY MOBILENETV2?
        - Lightweight: Fast inference (important for clinical use)
        - Pretrained on ImageNet: Good feature extractor
        - Efficient: Works well with limited data
        
        APPROACH:
        - Use MobileNetV2 as feature extractor (freeze weights)
        - Add custom classification head
        - Fine-tune if needed
        """
        print("[*]  Building Transfer Learning model (MobileNetV2)...")
        
        # Load pretrained MobileNetV2 (without top classification layer)
        base_model = MobileNetV2(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model weights for transfer learning
        base_model.trainable = not FREEZE_BASE_MODEL
        
        # Build model with stronger regularization
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION)),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION)),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(NUM_CLASSES if NUM_CLASSES > 2 else 1, 
                        activation='softmax' if NUM_CLASSES > 2 else 'sigmoid')
        ], name='FetalAbnormality_MobileNetV2')
        
        self.model = model
        print("[*] Transfer Learning model built successfully with L2 regularization")
        return model
    
    def build(self):
        """
        Build the model based on configuration
        """
        if self.use_transfer_learning:
            return self.build_transfer_learning_model()
        else:
            return self.build_custom_cnn()
    
    def compile_model(self, learning_rate=LEARNING_RATE):
        """
        Compile the model with optimizer, loss, and metrics
        
        LOSS FUNCTION: Categorical Crossentropy
        - Suitable for multi-class classification (Normal, Benign, Malignant)
        - Measures difference between predicted and true probabilities
        
        OPTIMIZER: Adam with Gradient Clipping
        - Adaptive learning rate
        - Gradient clipping prevents loss explosion
        - Works well for medical imaging
        - Faster convergence than SGD
        
        METRICS:
        - Accuracy: Overall correctness
        - Precision: Of predicted abnormal, how many are truly abnormal
        - Recall: Of actual abnormal, how many we detected
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        # Use Adam optimizer with gradient clipping to prevent loss explosion
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipvalue=1.0  # Clip gradients to [-1.0, 1.0]
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=LOSS_FUNCTION,
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        print("[*] Model compiled successfully")
    
    def summary(self):
        """
        Print model architecture summary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("=" * 70)
        self.model.summary()
        print("=" * 70)
        
        # Calculate parameters
        total_params = self.model.count_params()
        print(f"\n[*] Total Parameters: {total_params:,}")
        print(f"[*] Approximate Model Size: {total_params * 4 / (1024**2):.2f} MB")
    
    def get_model(self):
        """
        Return the Keras model
        """
        return self.model
    
    def save_model(self, filepath=GLOBAL_MODEL_PATH):
        """
        Save model to disk
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        self.model.save(filepath)
        print(f"[*] Model saved to: {filepath}")
    
    def load_model(self, filepath=GLOBAL_MODEL_PATH):
        """
        Load model from disk
        """
        self.model = keras.models.load_model(filepath)
        print(f"[*] Model loaded from: {filepath}")
        return self.model


def create_model():
    """
    Factory function to create and compile a model
    """
    cnn = FetalAbnormalityCNN()
    cnn.build()
    cnn.compile_model()
    return cnn.get_model()


def main():
    """
    Test model creation
    """
    print("=" * 70)
    print("[*] FETAL ABNORMALITY DETECTION - CNN MODEL")
    print("=" * 70)
    
    # Create model
    cnn = FetalAbnormalityCNN(use_transfer_learning=False)
    cnn.build()
    cnn.compile_model()
    cnn.summary()
    
    # Test with dummy data
    print("\n🧪 Testing model with dummy input...")
    dummy_input = tf.random.normal((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    output = cnn.get_model().predict(dummy_input, verbose=0)
    print(f"[*] Model output shape: {output.shape}")
    print(f"[*] Prediction value: {output[0][0]:.4f} (0=Normal, 1=Abnormal)")
    
    print("\n[*] Model creation successful!")


if __name__ == "__main__":
    main()
