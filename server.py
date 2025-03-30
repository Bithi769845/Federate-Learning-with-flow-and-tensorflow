import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from data_preprocessing import preprocess_data
from model import create_model
from hyperparameter_tuning import load_hyperparameters
from figure import plot_global_metrics

class MetricAggregator(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = []
        self.round_number = 0

    def evaluate_config(self, server_round: int, parameters, client_manager):
        """Return evaluation configuration dict for each round."""
        config = {"round": server_round}
        if server_round >= 0:  # Evaluate after every round
            return config
        return None

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        try:
            # Initialize metrics
            aggregated_metrics = {
                "loss": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc_roc": 0.0
            }
            
            total_examples = 0
            all_metrics = []

            # Validate metrics from each client
            for client_proxy, res in results:
                try:
                    metrics = res.metrics
                    required_metrics = ['accuracy', 'loss', 'f1', 'precision', 'recall', 'auc_roc']
                    missing_metrics = [m for m in required_metrics if m not in metrics]
                    if missing_metrics:
                        print(f"Warning: Client {client_proxy.cid} missing metrics: {missing_metrics}")
                        continue
                        
                    # Validate metric values
                    for metric, value in metrics.items():
                        if not isinstance(value, (int, float)) or np.isnan(value):
                            print(f"Warning: Invalid {metric} value from client {client_proxy.cid}")
                            continue
                    
                    # Add valid metrics
                    all_metrics.append((res.num_examples, metrics))
                    total_examples += res.num_examples
                        
                except Exception as e:
                    print(f"Error processing client {client_proxy.cid}: {e}")
                    continue

            if total_examples == 0:
                return None, aggregated_metrics

            # Calculate weighted metrics
            for metric in aggregated_metrics.keys():
                weighted_sum = 0.0
                for num_examples, metrics in all_metrics:
                    try:
                        value = float(metrics.get(metric, 0.0))
                        weighted_sum += value * (num_examples / total_examples)
                    except (TypeError, ValueError) as e:
                        print(f"Error with metric {metric}: {e}")
                aggregated_metrics[metric] = weighted_sum

            # Enhanced metrics display
            print(f"\n{'='*60}")
            print(f"Round {server_round} Global Metrics Summary")
            print(f"{'='*60}")
            for metric, value in aggregated_metrics.items():
                print(f"{metric.upper():15s}: {value:.4f}")
            print(f"{'='*60}\n")

            # Save metrics history
            self.metrics_history.append({
                "round": server_round,
                **aggregated_metrics
            })

            # Save and plot global metrics
            try:
                pd.DataFrame(self.metrics_history).to_csv('global_metrics.csv', index=False)
                plot_global_metrics(pd.DataFrame(self.metrics_history))
            except Exception as e:
                print(f"Error saving/plotting metrics: {e}")

            # Add connection verification
            print("\nSaving round results:")
            print(f"Global metrics saved to: global_metrics.csv")
            print(f"Visualizations saved to: figures/global/")
            print(f"Metrics from {len(results)} clients aggregated")

            return aggregated_metrics["loss"], aggregated_metrics

        except Exception as e:
            print(f"Error in aggregate_evaluate: {e}")
            return None, {}

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results from clients."""
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        print(f"\nCompleted training round {server_round}")
        print(f"Number of clients participated: {len(results)}")
        print(f"Number of failures: {len(failures)}")
        return aggregated_weights

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    if not metrics:
        return {}
    aggregated = {}
    for key in metrics[0][1].keys():
        if key == 'auc_roc':  # Handle AUC-ROC differently
            values = [m[1][key] for m in metrics]
            aggregated[key] = np.mean(values)
        else:
            total = sum(num_examples * m[1][key] for num_examples, m in metrics)
            aggregated[key] = total / sum(num_examples for num_examples, _ in metrics)
    return aggregated

def start_server():
    try:
        print("\nServer Data Flow Verification:")
        print("1. Loading data shapes from data_preprocessing.py")
        print("2. Loading hyperparameters from hyperparameter_tuning.py")
        print("3. Creating model from model.py")
        print("4. Saving metrics for figure.py")
        
        print("\nStarting Federated Learning Server...")
        print("Waiting for clients to connect...\n")
        
        # Get the correct input shape from data
        df, num_cols, _, _ = preprocess_data()
        input_shape = len(num_cols)
        print(f"Server model input shape: {input_shape}")
        
        # Load hyperparameters and create model
        best_hps = load_hyperparameters()
        print("\nUsing hyperparameters:", best_hps)
        
        # Server uses model.py to create initial global model
        initial_model = create_model(input_shape=input_shape, best_hps=best_hps)
        print("\nServer model architecture:")
        initial_model.summary()
        
        strategy = MetricAggregator(
            min_available_clients=4,
            min_fit_clients=4,
            min_evaluate_clients=4,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=weighted_average,
            initial_parameters=fl.common.ndarrays_to_parameters(initial_model.get_weights()),
            on_fit_config_fn=lambda r: {
                "batch_size": 64,  # Increased batch size
                "epochs": 5,       # Fewer epochs but more rounds
                "validation_split": 0.2
            }
        )
        
        server_config = fl.server.ServerConfig(
            num_rounds=15,  # More rounds
            round_timeout=1200.0
        )
        
        print("\nServer is ready! Waiting for clients...")
        print("To connect clients, open new terminals and run:")
        print("  python client.py 1")
        print("  python client.py 2")
        print("  python client.py 3")
        print("  python client.py 4")
        
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=server_config,
            strategy=strategy
        )
    except Exception as e:
        print(f"\nError starting server: {e}")
        print("\nPlease ensure:")
        print("1. Port 8080 is available")
        print("2. All dependencies are installed")
        print("3. Data preprocessing is complete")
        raise

if __name__ == "__main__":
    start_server()