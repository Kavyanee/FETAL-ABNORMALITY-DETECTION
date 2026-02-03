"""
ML Service Module

This module handles:
- Loading the trained model
- Preprocessing uploaded images
- Running inference
- Returning predictions

WHY THIS IS SEPARATE:
- Separation of concerns
- Easy to test
- Can be reused across different endpoints
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io


class MLService:
    """
    Machine Learning Service for Fetal Abnormality Detection
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize ML service
        
        Args:
            model_path: Path to trained model (.h5 file)
        """
        if model_path is None:
            # Default path
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'models',
                'global_model.h5'
            )
        
        self.model_path = model_path
        self.model = None
        self.img_height = 224  # Model expects 224x224 input
        self.img_width = 224   # Model expects 224x224 input
        self.class_names = ['Normal', 'Benign', 'Malignant']
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self):
        """
        Load the trained TensorFlow model
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Please train the model first using ml/train_federated.py"
            )
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"[OK] Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess uploaded image for model inference
        
        Steps:
        1. Convert bytes to image
        2. Resize to model input size
        3. Normalize pixel values to [0, 1]
        4. Add batch dimension
        
        Args:
            image_bytes: Raw image bytes from upload
        
        Returns:
            Preprocessed image array ready for model
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (in case of grayscale or RGBA)
        image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Resize to model input size
        image_array = cv2.resize(image_array, (self.img_width, self.img_height))
        
        # Normalize to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension (model expects batch)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, image_bytes: bytes) -> dict:
        """
        Run inference on uploaded image
        
        Args:
            image_bytes: Raw image bytes from upload
        
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise Exception("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_bytes)
        
        # Run inference
        prediction_proba = self.model.predict(processed_image, verbose=0)[0]
        
        # Debug: Print raw probabilities
        print(f"\n[DEBUG] Raw probabilities: {prediction_proba}")
        
        # Get predicted class
        prediction_class = int(np.argmax(prediction_proba))
        prediction_label = self.class_names[prediction_class]
        confidence = float(prediction_proba[prediction_class])
        
        print(f"[DEBUG] Predicted class: {prediction_class} ({prediction_label})")
        print(f"[DEBUG] Confidence: {confidence:.4f}\n")
        
        # Create probability dict for all classes
        probabilities = {self.class_names[i]: float(prediction_proba[i]) 
                        for i in range(len(self.class_names))}
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {'status': 'Model not loaded'}
        
        return {
            'status': 'Model loaded',
            'model_path': self.model_path,
            'input_shape': f"{self.img_height}x{self.img_width}x3",
            'classes': self.class_names,
            'total_parameters': self.model.count_params()
        }


# Global instance (singleton pattern)
ml_service = None


def get_ml_service() -> MLService:
    """
    Get or create ML service instance (singleton)
    
    WHY SINGLETON:
    - Model loading is expensive
    - Load once, reuse for all requests
    - Saves memory and time
    """
    global ml_service
    if ml_service is None:
        ml_service = MLService()
    return ml_service
