import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Tuple, Optional
from data_preprocessing import preprocess_data, create_non_iid_data
from hyperparameter_tuning import tune_hyperparameters, load_hyperparameters
from figure import analyze_results, plot_client_data_distribution
import os
import time
from model import create_model
import shutil

class FLClient(fl.client.NumPyClient): 
    def __init__(self, model, X_train, y_train, X_val, y_val, client_id, total_rounds):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.history = []
        self.client_id = client_id
        self.training_time = 0
        self.classes = ['Normal', 'Attack']  # Add this line
        self.total_rounds = total_rounds
        self.current_round = 0

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # print("parameter", parameters)
        self.model.set_weights(parameters)
        
        start_time = time.time()
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=config.get("epochs", 15),
            batch_size=config.get("batch_size", 256),
            validation_data=(self.X_val, self.y_val),
            verbose=0
        )
        self.training_time += time.time() - start_time

        print(f"Training time for client {self.client_id}: {self.training_time:.2f} seconds")
        
        # Calculate and store metrics
        y_pred = self.model.predict(self.X_val, verbose=0)
       
        if y_pred.shape[1] > 1:
         print("Warning: Multiclass output detected")
       
        y_pred_class = (y_pred >= 0.5).astype(int).flatten()

        round_metrics = {
            "loss": history.history['loss'][-1],
            "val_loss": history.history['val_loss'][-1],
            "accuracy": history.history['accuracy'][-1],
            "val_accuracy": history.history['val_accuracy'][-1],
            "f1": f1_score(self.y_val, y_pred_class, average='binary'),
            "auc_roc": roc_auc_score(self.y_val, y_pred.flatten())
        }
        self.history.append(round_metrics)

        # print("history" , self.history)
        
        # Print current round metrics
        print(f"\nClient {self.client_id} Round Metrics:")
        for k, v in round_metrics.items():
            print(f"{k}: {v:.4f}")
        # print("X_train Length:", len(self.X_train))
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        try:
            self.model.set_weights(parameters)
            
            loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
            print("accuracy and loss:", accuracy, loss)
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred_class = np.round(y_pred).flatten()
            y_score = y_pred.flatten()  # For binary classification
            # self.current_round = current_round
            self.current_round = config.get("current_round", self.current_round + 1)

            # Calculate metrics
            metrics = {
                "precision": precision_score(self.y_val, y_pred_class, average='binary'),
                "recall": recall_score(self.y_val, y_pred_class, average='binary'),
                "f1": f1_score(self.y_val, y_pred_class, average='binary'),
                "auc_roc": roc_auc_score(self.y_val, y_score),
                "accuracy": accuracy,
                "loss": loss
            }

            self._save_metrics(metrics, y_pred_class, y_score)
            
            # Generate visualizations
            history_df = pd.DataFrame(self.history)

            analyze_results(history_df, self.y_val, y_pred_class, y_score, self.classes, self.client_id)
            
            return loss, len(self.X_val), metrics
        except Exception as e:
            print(f"Error in evaluate: {e}")
            raise

    def _save_metrics(self, metrics, y_pred, y_score):
        """Save only the last round's predictions for all clients in a single file"""
        labels_file = 'fl_labels.csv'
        
        # Create DataFrame with current client's data
        current_data = pd.DataFrame({
            'client_id': [self.client_id] * len(self.y_val),
            'true_labels': self.y_val,
            'predicted_labels': y_pred,
            'scores': y_score
        })
        print("Current Round" , self.current_round)
        print("Total Rounds" , self.total_rounds)

        print(f"Saving data for client {self.client_id}")
        print(f"Data shape: {current_data.shape}")
        # Only save if it's the last round
        if self.current_round == self.total_rounds:
            try:
                if os.path.exists(labels_file):
                    existing_data = pd.read_csv(labels_file)
                    print(f"Existing data shape before update: {existing_data.shape}")
                    existing_data = existing_data[existing_data['client_id'] != self.client_id]
                    updated_data = pd.concat([existing_data, current_data])
                    print(f"Updated data shape: {updated_data.shape}")
                    updated_data.to_csv(labels_file, index=False)
                else:
                    current_data.to_csv(labels_file, index=False)
                    
                print(f"Successfully saved predictions for client {self.client_id} (Round {self.current_round})")
            except Exception as e:
                print(f"Error saving data for client {self.client_id}: {e}")

def main(client_id):
    print(f"\n{'='*20}")
    print(f"Starting Client {client_id}")
    print(f"{'='*20}\n")
    
    # Load and preprocess data
    df, num_cols, label_mapping, _ = preprocess_data()
    client_data, distribution = create_non_iid_data(df, num_cols)

   
    # Verify client_id is valid
    if client_id not in client_data:
        valid_ids = list(client_data.keys())
        print(f"Error: Invalid client_id {client_id}. Valid IDs are: {valid_ids}")
        return


    # Initialize model with correct input shape
    X_train, y_train, X_val, y_val = client_data[client_id]

    if len(np.unique(y_train)) > 2 or len(np.unique(y_val)) > 2:
     print(f"Warning: Client {client_id} may not be binary!")
    else :
        print(f"Client {client_id} is binary!")

    input_shape = len(num_cols)

    best_hps = load_hyperparameters()


    model = create_model(input_shape=input_shape, best_hps=best_hps)
    
    # Start client
    print(f"\nStarting Federated Learning for Client {client_id}")
    total_rounds = 10
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FLClient(model, X_train, y_train, X_val, y_val, client_id, total_rounds)
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        print("Example: python client.py 0")
        sys.exit(1)
    
    try:
        client_id = int(sys.argv[1])
        main(client_id)
    except ValueError:
        print("Error: client_id must be a number")
        sys.exit(1)