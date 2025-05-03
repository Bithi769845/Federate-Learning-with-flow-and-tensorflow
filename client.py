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

class FLClient(fl.client.NumPyClient): 
    def __init__(self, model, X_train, y_train, X_val, y_val, client_id):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.history = []
        self.client_id = client_id
        self.training_time = 0
        self.classes = ['Normal', 'Attack']  # Add this line
        # print(f"Client {client_id} initialized with {len(X_train)} training samples and {len(X_val)} validation samples")
        # print("model summary " ,self.model.summary())  # Print model summary for debugging
        # print("trainging " ,self.X_train.shape, self.y_train.shape, self.X_val.shape, self.y_val.shape)  # Print shapes for debugging
        # print("hostory" , self.history)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # print("parameter", parameters)
        self.model.set_weights(parameters)
        
        start_time = time.time()
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=config.get("epochs", 5),
            batch_size=config.get("batch_size", 256),
            validation_data=(self.X_val, self.y_val),
            verbose=0
        )
        self.training_time += time.time() - start_time
        
        # Calculate and store metrics
        y_pred = self.model.predict(self.X_val, verbose=0)
        # print("y val" , self.y_val)
        # # print("y_val ", self.X_val)  # Debugging line
        # print("y_pred:", y_pred)  # Debugging line
        # print("y_pred shape:", y_pred.shape)  # Debugging line
        if y_pred.shape[1] > 1:
         print("Warning: Multiclass output detected")
        # else:
        #  print("Binary classification output detected")
       #y_pred_class = np.round(y_pred).flatten()
        # print("y pred" , y_pred.shape)
        y_pred_class = (y_pred >= 0.5).astype(int).flatten()

        # print("y_pred_class" , y_pred_class.shape)
        # print

        # print("y_pred" , y_pred)

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
            # print("y_pred_class" , y_pred_class)
            # print("y_pred" , y_pred)
            # print("y_score" , y_score)
            # Calculate metrics
            metrics = {
                "precision": precision_score(self.y_val, y_pred_class, average='binary'),
                "recall": recall_score(self.y_val, y_pred_class, average='binary'),
                "f1": f1_score(self.y_val, y_pred_class, average='binary'),
                "auc_roc": roc_auc_score(self.y_val, y_score),
                "accuracy": accuracy,
                "loss": loss
            }
            # print(matrics)
            # print("metrics" , metrics)
            # Save client-specific metrics
            self._save_metrics(metrics, y_pred_class, y_score)
            
            # Generate visualizations
            history_df = pd.DataFrame(self.history)
            # print("y_true:", self.y_val)
            # print("y_pred_class:", y_pred_class)
            # print("history" , history_df)
            # history_df.to_csv(f'client_{self.client_id}_history.csv', index=False)
            # print(self.classes)
            # Initialize global lists to store the aggregated data
           
            
          
            

            
            # if(self.client_id is None):
            #     analyze_results(history_df, self.y_val, y_pred_class, y_score, self.classes)
            # else:
            analyze_results(history_df, self.y_val, y_pred_class, y_score, self.classes, self.client_id)
            
            return loss, len(self.X_val), metrics
        except Exception as e:
            print(f"Error in evaluate: {e}")
            raise

    def _save_metrics(self, metrics, y_pred, y_score):
        """Save metrics to client-specific files"""
        client_suffix = f"_client_{self.client_id}"
        # print("Y Val",self.y_val)
        # print("Y Pred" , y_pred)
        # print(" Y Score ", y_score)    
        # Save numerical metrics


        # pd.DataFrame([metrics]).to_csv(
        #     f'fl_metrics{client_suffix}.csv',
        #     mode='a',
        #     header=not os.path.exists(f'fl_metrics{client_suffix}.csv'),
        #     index=False
        # )

        # print("client suffix",client_suffix)
        
        # Save predictions
        # pd.DataFrame({
        #     'true_labels': self.y_val,
        #     'predicted_labels': y_pred,
        #     'scores': y_score
        # }).to_csv(
        #     f'fl_labels{client_suffix}.csv',
        #     mode='a',
        #     header=not os.path.exists(f'fl_labels{client_suffix}.csv'),
        #     index=False
        # )

def main(client_id):
    print(f"\n{'='*20}")
    print(f"Starting Client {client_id}")
    print(f"{'='*20}\n")
    
    # Load and preprocess data
    df, num_cols, label_mapping, _ = preprocess_data()
    client_data, distribution = create_non_iid_data(df, num_cols)

    # print(f"Client {client_id} data shape: {df.shape}")
    # print(f"Client {client_id} num_cols: {num_cols}\n")

    # print(f"Client {client_id} data shape: {client_data[client_id][0].shape}")
    # print(f"Client {client_id} data distribution: {distribution[client_id]}")   
    
    # Verify client_id is valid
    if client_id not in client_data:
        valid_ids = list(client_data.keys())
        print(f"Error: Invalid client_id {client_id}. Valid IDs are: {valid_ids}")
        return
    
    # print(f"Client {client_id} data shape: {client_data[client_id][0].shape}")

    # Initialize model with correct input shape
    X_train, y_train, X_val, y_val = client_data[client_id]

    # print(f"Client {client_id} data shape: X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")
    # 🔍 Debug: Check label values
    # print("Unique y_val labels:", np.unique(y_val))  # ✅ Add here

    if len(np.unique(y_train)) > 2 or len(np.unique(y_val)) > 2:
     print(f"Warning: Client {client_id} may not be binary!")
    else :
        print(f"Client {client_id} is binary!")

    input_shape = len(num_cols)
    # print(f"Client {client_id} model input shape: {input_shape}")
    
    # Only perform tuning if hyperparameters don't exist
    if client_id == 0 and not os.path.exists('best_hyperparameters.json'):
        # print(f"\nTuning hyperparameters (Client {client_id} is primary)...")
        best_hps = tune_hyperparameters()
    else:
        # print(f"\nClient {client_id} loading existing hyperparameters...")
        best_hps = load_hyperparameters()
    
    # print(f"Client {client_id} using hyperparameters:", best_hps)

    model = create_model(input_shape=input_shape, best_hps=best_hps)
    
    # print(f"\nClient {client_id} model architecture:")
    # model.summary()
    
    # Save initial client info
    client_info = {
        'client_id': client_id,
        'input_shape': input_shape,
        'attack_category': distribution[client_id]['attack_category'],
        'total_samples': len(X_train) + len(X_val)
    }

    # print(f"Client {client_id} info: {client_info}")
    
    # pd.DataFrame([client_info]).to_csv(
    #     f'client_info_{client_id}.csv',
    #     index=False
    # )
    
    # Start client with more verbose output
    # print(f"\nStarting Federated Learning for Client {client_id}")
    # print(f"Attack Category: {distribution[client_id]['attack_category']}")
    # print(f"Training samples: {len(X_train)}")
    # print(f"Validation samples: {len(X_val)}")
    
    # Start client
    print(f"\nStarting Federated Learning for Client {client_id}")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FLClient(model, X_train, y_train, X_val, y_val, client_id)
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