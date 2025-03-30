import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List
import os
from data_preprocessing import create_non_iid_data, preprocess_data
import pickle

def ensure_directory(directory="figures"):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_distribution():
    """Remove this function as it's redundant with plot_client_data_distribution"""
    pass  # We'll use plot_client_data_distribution instead

def plot_metrics(history: pd.DataFrame, client_id=None):
    ensure_directory()
    
    metrics = ["loss", "accuracy", "f1", "auc_roc"]
    titles = ["Loss vs Rounds", "Accuracy vs Rounds", "F1 & AUC-ROC Scores"]
    
    for metric, title in zip(metrics, titles):
        if metric in history.columns:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(history) + 1), history[metric].values, 
                    label=f"Training {metric.title()}", linestyle='-', marker='o')
            if f"val_{metric}" in history.columns:
                plt.plot(range(1, len(history) + 1), history[f"val_{metric}"].values,
                        label=f"Validation {metric.title()}", linestyle='--', marker='x')
            plt.title(title)
            plt.xlabel('Rounds')
            plt.ylabel(metric.title())
            plt.legend()
            plt.grid(True)
            if client_id is not None:
                plt.savefig(f"figures/{metric}_vs_rounds_client_{client_id}.png")
            else:
                plt.savefig(f"figures/{metric}_vs_rounds.png")
            plt.close()
        else:
            print(f"Metric {metric} not found in history")

def plot_confusion_matrix(y_true, y_pred, classes, filename):
    ensure_directory()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # For binary classification, use 2x2 matrix
    if len(classes) == 2:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"figures/{filename}")
    plt.close()

def plot_roc_curve(y_true, y_score, classes, filename):
    ensure_directory()
    plt.figure(figsize=(8, 6))
    
    # Simplified for binary classification
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/{filename}")
    plt.close()

def analyze_results(history: pd.DataFrame, y_true, y_pred, y_score, classes, client_id=None):
    try:
        # Set base directory
        if client_id is not None:
            base_dir = f"figures/client_{client_id}"
        else:
            base_dir = "figures/global"  # Changed from "figures" to "figures/global"
        ensure_directory(base_dir)

        # Print what we're analyzing
        print(f"\nAnalyzing {'global' if client_id is None else f'client {client_id}'} results...")
        
        # Generate plots
        # 1. Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Loss', marker='o', color='blue')
        plt.title(f"{'Global' if client_id is None else f'Client {client_id}'} Loss vs Rounds")
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{base_dir}/loss.png")
        plt.close()

        # 2. Accuracy Plot
        plt.figure(figsize=(10, 6))
        plt.plot(history['accuracy'], label='Accuracy', marker='o', color='green')
        plt.title(f"{'Global' if client_id is None else f'Client {client_id}'} Accuracy vs Rounds")
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{base_dir}/accuracy.png")
        plt.close()

        # 3. Confusion Matrix with percentages
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title(f"{'Global' if client_id is None else f'Client {client_id}'} Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Add accuracy text
        acc = accuracy_score(y_true, y_pred)
        plt.text(0.5, -0.1, f'Accuracy: {acc:.2%}', 
                horizontalalignment='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{base_dir}/confusion_matrix.png")
        plt.close()

        # 4. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{'Global' if client_id is None else f'Client {client_id}'} ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"{base_dir}/roc_curve.png")
        plt.close()

        # 5. Save detailed metrics
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='binary'),
            'Recall': recall_score(y_true, y_pred, average='binary'),
            'F1-Score': f1_score(y_true, y_pred, average='binary'),
            'AUC-ROC': roc_auc
        }
        
        # Print metrics
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save metrics to CSV
        pd.DataFrame([metrics]).to_csv(f"{base_dir}/metrics_summary.csv", index=False)

    except Exception as e:
        print(f"Error in analyze_results: {e}")
        import traceback
        print(traceback.format_exc())

def plot_client_data_distribution():
    """Plot data distribution using bar charts for normal vs attack samples"""
    ensure_directory()
    
    try:
        # Load distribution information
        with open("distribution_info.pkl", "rb") as f:
            distribution_info = pickle.load(f)
        
        # Extract data
        clients = sorted(distribution_info.keys())
        attack_categories = [distribution_info[c]['attack_category'] for c in clients]
        normal_samples = [distribution_info[c]['normal_samples'] for c in clients]
        attack_samples = [distribution_info[c]['attack_samples'] for c in clients]
        
        # Create bar plot
        bar_width = 0.35
        x = np.arange(len(clients))
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - bar_width/2, normal_samples, bar_width, label="Normal Traffic", color='skyblue')
        plt.bar(x + bar_width/2, attack_samples, bar_width, label="Attack Traffic", color='salmon')
        
        plt.xlabel("Client ID")
        plt.ylabel("Number of Samples")
        plt.title("Data Distribution Across Clients")
        plt.xticks(x, [f"Client {c}\n({cat})" for c, cat in zip(clients, attack_categories)], rotation=15)
        plt.legend()
        
        # Add value labels on top of bars
        for i in range(len(clients)):
            plt.text(i - bar_width/2, normal_samples[i], str(normal_samples[i]), 
                    ha='center', va='bottom')
            plt.text(i + bar_width/2, attack_samples[i], str(attack_samples[i]), 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("figures/client_data_distribution.png")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting distribution: {e}")

def plot_client_accuracy_distribution(client_accuracies: Dict[int, float]):
    ensure_directory()
    plt.figure(figsize=(10, 6))
    plt.bar(client_accuracies.keys(), client_accuracies.values(), color='skyblue')
    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.title('Client Accuracy Distribution')
    plt.grid(True)
    plt.savefig("figures/client_accuracy_distribution.png")
    plt.close()

def plot_training_time_vs_clients(training_times: Dict[int, float]):
    ensure_directory()
    plt.figure(figsize=(10, 6))
    plt.bar(training_times.keys(), training_times.values(), color='lightgreen')
    plt.xlabel('Client ID')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Clients')
    plt.grid(True)
    plt.savefig("figures/training_time_vs_clients.png")
    plt.close()

def plot_all_clients_distribution():
    ensure_directory()
    
    try:
        with open("distribution_info.pkl", "rb") as f:
            distribution_info = pickle.load(f)
        
        num_clients = len(distribution_info)
        if num_clients == 0:
            print("No client data available")
            return
            
        # Create subplots for each client
        fig, axes = plt.subplots(1, num_clients, figsize=(6*num_clients, 5))
        fig.suptitle('Data Distribution Across All Clients', fontsize=16, y=1.05)
        
        # Ensure axes is always an array
        if num_clients == 1:
            axes = [axes]
            
        # Plot distribution for each client
        for client_id in range(1, num_clients + 1):
            if client_id in distribution_info:
                info = distribution_info[client_id]
                sizes = [info['normal_samples'], info['attack_samples']]
                labels = ['Normal', info['attack_category']]
                axes[client_id-1].pie(sizes, labels=labels, autopct='%1.1f%%')
                axes[client_id-1].set_title(f'Client {client_id}\n({info["attack_category"]})')
        
        plt.tight_layout()
        plt.savefig("figures/all_clients_distribution.png")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting distribution: {e}")

def plot_global_metrics(history: pd.DataFrame):
    """Plot metrics from server's global results"""
    print("\nFigure Generation Verification:")
    print("1. Reading metrics from server")
    print("2. Reading client metrics")
    print("3. Generating visualizations")
    print(f"4. Saving to figures/global/ directory")
    
    ensure_directory("figures/global")
    
    try:
        # Plot individual metrics
        metrics = ["accuracy", "loss", "f1", "auc_roc", "precision", "recall"]
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(history[metric], 'b-', marker='o', linewidth=2)
            plt.title(f'Global {metric.upper()} vs Rounds')
            plt.xlabel('Rounds')
            plt.ylabel(metric.title())
            plt.grid(True)
            plt.savefig(f"figures/global/{metric}_history.png")
            plt.close()
        
        # Plot combined metrics
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            plt.plot(history[metric], 'b-', marker='o')
            plt.title(f'Global {metric.upper()}')
            plt.xlabel('Rounds')
            plt.ylabel(metric.title())
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("figures/global/all_metrics.png")
        plt.close()
        
        # Print final performance
        final_metrics = history.iloc[-1].to_dict()
        print("\nCurrent Global Performance:")
        print("=" * 40)
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{metric:15s}: {value:.4f}")
        print("=" * 40)
        
        pd.DataFrame([final_metrics]).to_csv("figures/global/final_metrics.csv")
        
    except Exception as e:
        print(f"Error in plot_global_metrics: {e}")

def plot_client_metrics(client_id: int):
    """Plot metrics from individual client results"""
    metrics_file = f'fl_metrics_client_{client_id}.csv'
    if os.path.exists(metrics_file):
        history = pd.read_csv(metrics_file)
        ensure_directory(f"figures/client_{client_id}")
        try:
            metrics = ["accuracy", "loss", "f1", "auc_roc", "precision", "recall"]
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                plt.plot(history[metric], 'b-', marker='o', linewidth=2)
                plt.title(f'Client {client_id} {metric.upper()} vs Rounds')
                plt.xlabel('Rounds')
                plt.ylabel(metric.title())
                plt.grid(True)
                plt.savefig(f"figures/client_{client_id}/{metric}_history.png")
                plt.close()
        except Exception as e:
            print(f"Error in plot_client_metrics for Client {client_id}: {e}")

if __name__ == "__main__":
    print("Loading real data for visualization...")
    
    try:
        # Create base directories
        ensure_directory("figures")
        ensure_directory("figures/global")
        
        # Try to load global metrics
        global_results_available = False
        if os.path.exists('global_metrics.csv'):
            global_history = pd.read_csv('global_metrics.csv')
            if os.path.exists('global_predictions.csv'):
                global_predictions = pd.read_csv('global_predictions.csv')
                global_results_available = True
                print("Generating global visualizations...")
                analyze_results(
                    global_history,
                    global_predictions['true_labels'],
                    global_predictions['predicted_labels'],
                    global_predictions['scores'],
                    ['Normal', 'Attack']
                )
            else:
                print("Global predictions file not found, skipping global analysis")
        else:
            print("Global metrics file not found, skipping global analysis")
        
        # Ensure global metrics are plotted
        if os.path.exists('global_metrics.csv'):
            print("\nGenerating global metrics visualizations...")
            global_history = pd.read_csv('global_metrics.csv')
            plot_global_metrics(global_history)
        
        # Generate client-specific visualizations
        client_results_available = False
        for client_id in range(1, 5):
            print(f"\nChecking Client {client_id} data...")
            ensure_directory(f"figures/client_{client_id}")
            
            metrics_file = f'fl_metrics_client_{client_id}.csv'
            labels_file = f'fl_labels_client_{client_id}.csv'
            
            if os.path.exists(metrics_file) and os.path.exists(labels_file):
                print(f"Generating visualizations for Client {client_id}")
                history = pd.read_csv(metrics_file)
                predictions = pd.read_csv(labels_file)
                
                analyze_results(
                    history,
                    predictions['true_labels'],
                    predictions['predicted_labels'],
                    predictions['scores'],
                    ['Normal', 'Attack'],
                    client_id
                )
                client_results_available = True
            else:
                print(f"Data files missing for Client {client_id}, skipping")
        
        # Always generate distribution plots
        print("\nGenerating distribution plots...")
        plot_client_data_distribution()
        plot_all_clients_distribution()
        
        if not global_results_available and not client_results_available:
            print("\nWarning: No results files found. Run training first!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        print(traceback.format_exc())
    
    print("\nVisualization generation complete!")
