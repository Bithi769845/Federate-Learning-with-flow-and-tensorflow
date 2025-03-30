import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from typing import Dict, Tuple, Optional
from data_preprocessing import preprocess_data, create_non_iid_data
from hyperparameter_tuning import tune_hyperparameters, load_hyperparameters
from figure import analyze_results, plot_client_data_distribution
import os
import time
import json
from model import create_model

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val, client_id):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.history = []
        self.client_id = client_id
        self.training_time = 0
        self.classes = ['Normal', 'Attack']  # Changed from 5 classes to binary

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Data augmentation for robustness
        train_data = self.X_train.copy()
        train_labels = self.y_train.copy()
        
        # Add Gaussian noise to 20% of training data
        noise_idx = np.random.choice(len(train_data), size=int(0.2*len(train_data)), replace=False)
        train_data[noise_idx] += np.random.normal(0, 0.01, train_data[noise_idx].shape)
        
        history = self.model.fit(
            train_data, train_labels,
            epochs=config.get("epochs", 8),
            batch_size=config.get("batch_size", 32),
            validation_split=config.get("validation_split", 0.2),
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        
        self.training_time += time.time() - time.time()
        
        # Calculate and store metrics
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_class = np.round(y_pred).flatten()
        
        round_metrics = {
            "loss": history.history['loss'][-1],
            "val_loss": history.history['val_loss'][-1],
            "accuracy": history.history['accuracy'][-1],
            "val_accuracy": history.history['val_accuracy'][-1],
            "f1": f1_score(self.y_val, y_pred_class, average='binary'),
            "auc_roc": roc_auc_score(self.y_val, y_pred.flatten())
        }
        self.history.append(round_metrics)
        
        # Print current round metrics
        print(f"\nClient {self.client_id} Round Metrics:")
        for k, v in round_metrics.items():
            print(f"{k}: {v:.4f}")
        
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        try:
            self.model.set_weights(parameters)
            
            # Get predictions first
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = np.asarray(y_pred).reshape(-1)
            y_pred_class = (y_pred > 0.5).astype(int)
            
            # Calculate all metrics
            metrics = {
                "loss": float(self.model.evaluate(self.X_val, self.y_val, verbose=0)[0]),
                "accuracy": float(accuracy_score(self.y_val, y_pred_class)),
                "precision": float(precision_score(self.y_val, y_pred_class, average='binary')),
                "recall": float(recall_score(self.y_val, y_pred_class, average='binary')),
                "f1": float(f1_score(self.y_val, y_pred_class, average='binary')),
                "auc_roc": float(roc_auc_score(self.y_val, y_pred))
            }
            
            # Add predictions for saving
            predictions = {
                "true_labels": self.y_val.tolist(),
                "predicted_labels": y_pred_class.tolist(),
                "scores": y_pred.tolist()
            }
            
            # Save metrics and predictions
            self._save_metrics(metrics, predictions)
            
            # Print evaluation metrics
            print(f"\nClient {self.client_id} Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            return metrics["loss"], len(self.X_val), metrics
            
        except Exception as e:
            print(f"Error in evaluate: {e}")
            return float('inf'), 0, {}

    def _save_metrics(self, metrics, predictions=None):
        """Save metrics to client-specific files"""
        client_suffix = f"_client_{self.client_id}"
        
        # Save numerical metrics
        metrics_df = pd.DataFrame([{k: v for k, v in metrics.items() if isinstance(v, (int, float))}])
        metrics_df.to_csv(
            f'fl_metrics{client_suffix}.csv',
            mode='a',
            header=not os.path.exists(f'fl_metrics{client_suffix}.csv'),
            index=False
        )
        
        # Save predictions
        if predictions is not None:
            pd.DataFrame({
                'true_labels': predictions['true_labels'],
                'predicted_labels': predictions['predicted_labels'],
                'scores': predictions['scores']
            }).to_csv(
                f'fl_labels{client_suffix}.csv',
                mode='w',  # Overwrite mode for predictions
                index=False
            )

def main(client_id):
    try:
        print("\nClient Data Flow Verification:")
        print("1. Loading data from data_preprocessing.py")
        print("2. Checking hyperparameters from hyperparameter_tuning.py")
        print("3. Creating model from model.py")
        print("4. Saving metrics for figure.py")
        
        print(f"\n{'='*50}")
        print(f"Starting Client {client_id}")
        print(f"{'='*50}\n")
        
        # Add server connection check
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 8080))
            sock.close()
            if result != 0:
                print("Error: Server is not running! Please start the server first.")
                print("\nTo start the system correctly:")
                print("1. Run server:     python server.py")
                print("2. Run clients:    python client.py <client_id>")
                return
        except Exception as e:
            print(f"Error checking server: {e}")
            return

        # Load and preprocess data
        df, num_cols, label_mapping, _ = preprocess_data()
        client_data, distribution = create_non_iid_data(df, num_cols, label_mapping)
        
        # Verify client_id is valid
        if client_id not in client_data:
            valid_ids = list(client_data.keys())
            print(f"Error: Invalid client_id {client_id}. Valid IDs are: {valid_ids}")
            return
        
        # Initialize model with correct input shape
        X_train, y_train, X_val, y_val = client_data[client_id]
        input_shape = len(num_cols)
        print(f"Client {client_id} model input shape: {input_shape}")
        
        # Verify tuning info if client is primary
        if client_id == 1:
            if os.path.exists('tuning_info.json'):
                with open('tuning_info.json', 'r') as f:
                    tuning_info = json.load(f)
                print("\nVerifying tuning information:")
                print(f"Input shape: {tuning_info['input_shape']}")
                print(f"Training samples: {tuning_info['training_samples']}")
                print(f"Tuning accuracy: {tuning_info['tuning_accuracy']:.4f}")
            else:
                print("\nNo tuning info found - will perform tuning")
        
        # Only perform tuning if hyperparameters don't exist
        if client_id == 1 and not os.path.exists('best_hyperparameters.json'):
            print(f"\nTuning hyperparameters (Client {client_id} is primary)...")
            best_hps = tune_hyperparameters()
            verify_tuning_results(best_hps)
        else:
            print(f"\nClient {client_id} loading existing hyperparameters...")
            best_hps = load_hyperparameters()
            
        if best_hps is None:
            raise ValueError("Failed to load or tune hyperparameters")
            
        print(f"Client {client_id} using hyperparameters:", best_hps)
        
        # Client uses model.py to create local model
        model = create_model(input_shape=input_shape, best_hps=best_hps)
        
        print(f"\nClient {client_id} model architecture:")
        model.summary()
        
        # Save initial client info
        client_info = {
            'client_id': client_id,
            'input_shape': input_shape,
            'attack_category': distribution[client_id]['attack_category'],
            'total_samples': len(X_train) + len(X_val)
        }
        
        pd.DataFrame([client_info]).to_csv(
            f'client_info_{client_id}.csv',
            index=False
        )
        
        # Start client with more verbose output
        print(f"\nStarting Federated Learning for Client {client_id}")
        print(f"Attack Category: {distribution[client_id]['attack_category']}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Start client
        print(f"\nStarting Federated Learning for Client {client_id}")
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=FLClient(model, X_train, y_train, X_val, y_val, client_id)
        )
    except Exception as e:
        print(f"\nError in client {client_id}: {e}")
        print("\nPlease ensure:")
        print("1. Server is running (python server.py)")
        print("2. Valid client ID is provided (1-4)")
        print("3. All dependencies are installed")

def verify_tuning_results(best_hps):
    """Verify tuning results are valid"""
    required_params = ['units_1', 'units_2', 'dropout_1', 'dropout_2', 'learning_rate']
    missing = [p for p in required_params if p not in best_hps]
    if missing:
        raise ValueError(f"Missing hyperparameters: {missing}")
    
    print("\nTuning verification:")
    print(f"Accuracy achieved: {best_hps.get('tuning_accuracy', 0):.4f}")
    for param, value in best_hps.items():
        if param != 'tuning_accuracy':
            print(f"{param:15s}: {value}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("\nError: Client ID is required!")
        print("\nUsage:")
        print("  python client.py <client_id>")
        print("\nAvailable client IDs:")
        print("  1 - DoS attacks")
        print("  2 - Recon attacks")
        print("  3 - MQTT attacks")
        print("  4 - MITM attacks")
        print("\nExample:")
        print("  python client.py 1")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        if client_id not in [1, 2, 3, 4]:
            print(f"\nError: Invalid client ID: {client_id}")
            print("Please use client IDs: 1, 2, 3, or 4")
            sys.exit(1)
        main(client_id)
    except ValueError:
        print("\nError: Client ID must be a number between 1 and 4")
        sys.exit(1)