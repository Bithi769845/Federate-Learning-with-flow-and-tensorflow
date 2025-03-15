import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from data_preprocessing import preprocess_data
from model import create_model
from hyperparameter_tuning import load_hyperparameters

class MetricAggregator(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = []
        self.round_number = 0

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated:
            loss, metrics = aggregated
            self.round_number += 1
            metrics['round'] = self.round_number
            self.metrics_history.append(metrics)
            print(f"\nRound {self.round_number} Metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
        return aggregated

    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        return aggregated_weights

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    aggregated = {}
    for key in metrics[0][1].keys():
        if key == 'auc_roc':  # Handle AUC-ROC differently
            values = [m[key] for _, m in metrics]
            aggregated[key] = np.mean(values)
        else:
            total = sum(num_examples * m[key] for num_examples, m in metrics)
            aggregated[key] = total / sum(num_examples for num_examples, _ in metrics)
    return aggregated

def start_server():
    try:
        # Get the correct input shape from data
        df, num_cols, _, _ = preprocess_data()
        input_shape = len(num_cols)
        print(f"Server model input shape: {input_shape}")
        
        # Load the same hyperparameters used by clients
        best_hps = load_hyperparameters()
        print("Using hyperparameters:", best_hps)
        
        # Initialize model with correct input shape and hyperparameters
        initial_model = create_model(input_shape=input_shape, best_hps=best_hps)
        print("Server model summary:")
        initial_model.summary()
        
        strategy = MetricAggregator(
            evaluate_metrics_aggregation_fn=weighted_average,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=1,  # Added
            min_fit_clients=1,        # Added
            initial_parameters=fl.common.ndarrays_to_parameters(initial_model.get_weights()),
        )
        
        # Add server config
        server_config = fl.server.ServerConfig(
            num_rounds=10,
            round_timeout=600.0  # 10 minutes timeout per round
        )
        
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=server_config,
            strategy=strategy
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        raise

if __name__ == "__main__":
    start_server()