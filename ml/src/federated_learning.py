"""
Federated Learning Implementation using Flower

WHAT IS FEDERATED LEARNING?
Traditional ML: All data → Central server → Train model
Federated ML: Data stays local → Train locally → Share only weights

WHY FEDERATED LEARNING FOR MEDICAL DATA?
1. Privacy: Patient data never leaves hospital
2. Compliance: Meets HIPAA, GDPR requirements
3. Collaboration: Hospitals can collaborate without sharing data
4. Better Models: Learn from diverse data sources

HOW IT WORKS:
1. Central server initializes global model
2. Server sends model to all hospitals
3. Each hospital trains on local data
4. Hospitals send weight updates back
5. Server aggregates weights (FedAvg algorithm)
6. Repeat for multiple rounds

FEDERATED AVERAGING (FedAvg):
- Weighted average of client models
- Weight = proportion of data at each client
- Mathematically: w_global = Σ(n_i/n_total * w_i)
"""

import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
import os
import gc

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.cnn_model import create_model
from src.data_preprocessing import DataPreprocessor


class FederatedClient(fl.client.NumPyClient):
    """
    Federated Learning Client (represents one hospital)
    
    Each client:
    - Has its own local dataset (loaded via tf.data.Dataset for memory efficiency)
    - Trains the model locally
    - Sends only model weights to server
    - Never shares raw patient data
    """
    
    def __init__(self, client_id: int, model: tf.keras.Model, 
                 train_dataset, val_dataset, X_val_data, y_val_data, class_weights=None):
        self.client_id = client_id
        self.model = model
        self.train_dataset = train_dataset  # tf.data.Dataset to load data on-the-fly
        self.val_dataset = val_dataset
        self.X_val = X_val_data  # Small validation set for evaluation
        self.y_val = y_val_data
        self.class_weights = class_weights  # Handle class imbalance
        
        # Count samples in dataset
        train_count = sum(1 for _ in train_dataset)
        val_count = sum(1 for _ in val_dataset)
        
        print(f"[*] Hospital {client_id} initialized:")
        print(f"   Training batches: {train_count}")
        print(f"   Validation batches: {val_count}")
        if class_weights is not None:
            print(f"   Class weights: {class_weights}")
    
    def get_parameters(self, config):
        """
        Return current model weights
        
        WHY: Server needs initial weights to start federated training
        """
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        """
        Train model on local data using tf.data.Dataset
        
        PROCESS:
        1. Receive global model weights from server
        2. Update local model with these weights
        3. Train on local hospital data (loaded via dataset)
        4. Return updated weights to server
        
        PRIVACY: Only weights are shared, not data!
        MEMORY: Uses tf.data.Dataset to avoid loading entire dataset into memory
        REGULARIZATION: Uses class weights and early stopping to prevent overfitting
        """
        # Update local model with global weights
        self.model.set_weights(parameters)
        
        # Clear session to free memory
        tf.keras.backend.clear_session()
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping to prevent overfitting
        if EARLY_STOPPING_ENABLED:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ))
        
        # Train locally using dataset
        print(f"\n[*] Hospital {self.client_id} training locally...")
        history = self.model.fit(
            self.train_dataset,
            epochs=LOCAL_EPOCHS,
            validation_data=self.val_dataset,
            verbose=1,
            class_weight=self.class_weights,  # Handle class imbalance
            callbacks=callbacks  # Add early stopping
        )
        
        # Return updated weights and training info
        # Count samples for weighted averaging
        num_samples = sum(len(batch[1]) for batch in self.train_dataset)
        
        return self.model.get_weights(), num_samples, {
            "train_loss": float(history.history['loss'][-1]),
            "train_accuracy": float(history.history['accuracy'][-1])
        }
    
    def evaluate(self, parameters, config):
        """
        Evaluate model on local validation data
        
        WHY: Track performance on each hospital's data
        """
        self.model.set_weights(parameters)
        loss, accuracy, precision, recall, auc = self.model.evaluate(
            self.X_val, self.y_val, verbose=0
        )
        
        print(f"[*] Hospital {self.client_id} evaluation:")
        print(f"   Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, len(self.X_val), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc)
        }


def create_client(client_id: int):
    """
    Factory function to create a federated client with memory-efficient data loading
    and automatic class weight computation
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Load client data using tf.data.Dataset
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, X_val_data, y_val_data = preprocessor.load_client_data(client_id)
    
    # Compute class weights to handle imbalance (Malignant >> Normal, Benign)
    if USE_CLASS_WEIGHTS and AUTOMATIC_CLASS_WEIGHTS:
        # Get class labels from validation data (as proxy for full distribution)
        class_labels = np.argmax(y_val_data, axis=1)
        class_weights_array = compute_class_weight(
            'balanced', 
            classes=np.unique(class_labels),
            y=class_labels
        )
        # Convert to dictionary {0: weight0, 1: weight1, 2: weight2}
        class_weights = {i: w for i, w in enumerate(class_weights_array)}
        print(f"[*] Computed class weights for Hospital {client_id}: {class_weights}")
    else:
        class_weights = None
    
    # Create model
    model = create_model()
    
    # Create client with class weights
    client = FederatedClient(client_id, model, train_dataset, val_dataset, 
                            X_val_data, y_val_data, class_weights=class_weights)
    
    return client


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Aggregate metrics from multiple clients using weighted average
    
    WHY: Clients with more data should have more influence
    
    FORMULA: metric_avg = Σ(num_examples_i * metric_i) / Σ(num_examples_i)
    """
    # Calculate total examples
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get metric names from first client
    metric_names = metrics[0][1].keys()
    
    # Weighted average for each metric
    for metric_name in metric_names:
        weighted_sum = sum([
            num_examples * m[metric_name] 
            for num_examples, m in metrics
        ])
        aggregated[metric_name] = weighted_sum / total_examples
    
    return aggregated


def get_evaluate_fn(model: tf.keras.Model):
    """
    Return an evaluation function for the server
    
    WHY: Server needs to evaluate global model on test set
    """
    # Load test data (we'll use hospital 0's validation set as test)
    preprocessor = DataPreprocessor()
    _, _, X_test, y_test = preprocessor.load_client_data(0)
    
    def evaluate(server_round: int, parameters, config):
        """
        Evaluate global model
        """
        model.set_weights(parameters)
        loss, accuracy, precision, recall, auc = model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"\n[*] Round {server_round} - Global Model Evaluation:")
        print(f"   Loss: {loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   AUC: {auc:.4f}")
        
        return loss, {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc)
        }
    
    return evaluate


class FederatedLearningServer:
    """
    Federated Learning Server (Central Aggregator)
    
    Responsibilities:
    - Initialize global model
    - Distribute model to clients
    - Aggregate client updates
    - Evaluate global model
    """
    
    def __init__(self, num_clients=NUM_CLIENTS, num_rounds=FL_ROUNDS):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.global_model = None
        
    def start_training(self):
        """
        Start federated learning training
        
        PROCESS:
        1. Initialize global model
        2. Configure federated strategy (FedAvg)
        3. Start Flower server
        4. Clients connect and train
        5. Server aggregates and updates global model
        6. Repeat for num_rounds
        """
        print("=" * 70)
        print("[*] STARTING FEDERATED LEARNING SERVER")
        print("=" * 70)
        print(f"Number of hospitals: {self.num_clients}")
        print(f"Federated rounds: {self.num_rounds}")
        print(f"Local epochs per round: {LOCAL_EPOCHS}")
        print("=" * 70)
        
        # Create initial global model
        self.global_model = create_model()
        
        # Define federated averaging strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Use all clients for training
            fraction_evaluate=1.0,  # Use all clients for evaluation
            min_fit_clients=self.num_clients,  # Minimum clients for training
            min_evaluate_clients=self.num_clients,  # Minimum clients for evaluation
            min_available_clients=self.num_clients,  # Wait for all clients
            evaluate_fn=get_evaluate_fn(self.global_model),  # Server-side evaluation
            fit_metrics_aggregation_fn=weighted_average,  # Aggregate training metrics
            evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate eval metrics
        )
        
        # Start Flower server
        print("\n[*] Starting Flower server...")
        print("⏳ Waiting for clients to connect...\n")
        
        # Note: In simulation mode, we'll use a different approach
        # For now, we'll implement a simulation-based training
        self._simulate_federated_training()
    
    def _simulate_federated_training(self):
        """
        Simulate federated learning without actual client-server communication
        
        WHY: Easier for demo and local testing
        NOTE: In production, use actual Flower server with fl.server.start_server()
        """
        print("[*] Running in SIMULATION mode (all clients on same machine)\n")
        
        # Create all clients
        clients = []
        for client_id in range(self.num_clients):
            client = create_client(client_id)
            clients.append(client)
        
        # Initialize global model
        global_weights = self.global_model.get_weights()
        
        # Federated training rounds
        for round_num in range(1, self.num_rounds + 1):
            print("\n" + "=" * 70)
            print(f"[*] FEDERATED ROUND {round_num}/{self.num_rounds}")
            print("=" * 70)
            
            # Store client updates
            client_weights = []
            client_sizes = []
            
            # Each client trains locally
            for client in clients:
                # Client trains and returns weights
                weights, num_examples, metrics = client.fit(global_weights, {})
                client_weights.append(weights)
                client_sizes.append(num_examples)
                
                print(f"   Hospital {client.client_id}: "
                      f"Loss={metrics['train_loss']:.4f}, "
                      f"Acc={metrics['train_accuracy']:.4f}")
                
                # Clear memory after each client
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
            
            # Aggregate weights (FedAvg)
            print("\n[*] Aggregating weights from all hospitals...")
            global_weights = self._federated_averaging(client_weights, client_sizes)
            self.global_model.set_weights(global_weights)
            
            # Evaluate global model
            print("\n[*] Evaluating global model...")
            for client in clients:
                loss, num_examples, metrics = client.evaluate(global_weights, {})
        
        print("\n" + "=" * 70)
        print("[*] FEDERATED LEARNING COMPLETE")
        print("=" * 70)
        
        # Save final global model
        self.save_global_model()
    
    def _federated_averaging(self, client_weights, client_sizes):
        """
        Implement Federated Averaging (FedAvg) algorithm
        
        FORMULA: w_global = Σ(n_i/n_total * w_i)
        
        WHERE:
        - w_global: Global model weights
        - w_i: Weights from client i
        - n_i: Number of samples at client i
        - n_total: Total samples across all clients
        """
        total_size = sum(client_sizes)
        
        # Initialize averaged weights
        avg_weights = []
        
        # For each layer's weights
        for layer_idx in range(len(client_weights[0])):
            # Weighted sum of this layer's weights from all clients
            layer_avg = np.zeros_like(client_weights[0][layer_idx])
            
            for client_idx, client_weight in enumerate(client_weights):
                weight = client_sizes[client_idx] / total_size
                layer_avg += weight * client_weight[layer_idx]
            
            avg_weights.append(layer_avg)
        
        return avg_weights
    
    def save_global_model(self, filepath=GLOBAL_MODEL_PATH):
        """
        Save the final global model
        """
        self.global_model.save(filepath)
        print(f"\n[*] Global model saved to: {filepath}")
        print("   This model can now be used for inference in the backend!")


def main():
    """
    Main function to run federated learning
    """
    print("=" * 70)
    print("[*] PRIVACY-PRESERVING FETAL ABNORMALITY DETECTION")
    print("   Federated Learning Training")
    print("=" * 70)
    
    # Create and start server
    server = FederatedLearningServer(
        num_clients=NUM_CLIENTS,
        num_rounds=FL_ROUNDS
    )
    
    server.start_training()
    
    print("\n[*] Training complete!")
    print(f"[*] Model saved to: {GLOBAL_MODEL_PATH}")
    print("\n[*] Next steps:")
    print("   1. Evaluate model performance (run evaluation.py)")
    print("   2. Deploy model to FastAPI backend")
    print("   3. Test with frontend UI")


if __name__ == "__main__":
    main()
