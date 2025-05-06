import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List
import os
from data_preprocessing import create_non_iid_data, preprocess_data
import pickle
from server import weighted_average as wa
from hyperparameter_tuning import tune_hyperparameters, load_hyperparameters
from model import create_model


def ensure_directory(directory="figures"):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_distribution():
    df, num_cols, label_mapping, label_encoders = preprocess_data()
    _, distribution = create_non_iid_data(df, num_cols, label_mapping)
    
    labels = ['Normal', 'Attack']
    client_data = [[dist.get(label.lower(), 0) for label in labels] for dist in distribution.values()]
    
    plt.figure(figsize=(8, 6))
    plt.pie(client_data[0], labels=labels, autopct='%1.1f%%')
    plt.title("Client Data Distribution")
    plt.savefig("figures/client_data_distribution.png")
    plt.close()

def plot_metrics(history: pd.DataFrame, client_id=None):
    ensure_directory()
    
    metrics = ["loss", "accuracy"]
    titles = ["Loss vs Rounds", "Accuracy vs Rounds"]
    
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
    cm = confusion_matrix(y_true, y_pred,labels=[0,1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"figures/{filename}")
    plt.close()

def plot_roc_curve(y_true, y_score, classes, filename):
    ensure_directory()
    plt.figure(figsize=(8, 6))
    
    # Handle binary classification case
    if y_score.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    else:
        # Multi-class case
        print('Hudai')
        # for i, cls in enumerate(classes):
        #     fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
        #     auc_score = auc(fpr, tpr)
        #     plt.plot(fpr, tpr, label=f'{cls} (AUC = {auc_score:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/{filename}")
    plt.close()

def analyze_results(history: pd.DataFrame, y_true, y_pred, y_score, classes, client_id=None):
    ensure_directory()
    try:
        if client_id is None:
            # Global federation results
            print("Generating global metrics plots...")
            plot_global_metrics(history)
            plot_confusion_matrix(y_true, y_pred, ['Normal', 'Attack'], "confusion_global.png")
            plot_roc_curve(y_true, y_score, ['Normal', 'Attack'], "roc_global.png")
            plot_all_clients_distribution()
        else:
            # Client-specific metrics
            print(f"Generating plots for Client {client_id}...")
            plot_metrics(history, client_id)
            plot_confusion_matrix(y_true, y_pred, ['Normal', 'Attack'], f"confusion_client_{client_id}.png")
            plot_roc_curve(y_true, y_score, ['Normal', 'Attack'], f"roc_client_{client_id}.png")

    except Exception as e:
        print(f"Error in analyze_results: {e}")

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
    """Plot data distribution for all clients in a single figure"""
    ensure_directory()
    
    try:
        with open("distribution_info.pkl", "rb") as f:
            distribution_info = pickle.load(f)
        
        num_clients = len(distribution_info)
        if num_clients == 0:
            print("No client data available")
            return
            
        fig, axes = plt.subplots(1, num_clients, figsize=(6*num_clients, 5))
        fig.suptitle('Data Distribution Across All Clients', fontsize=16, y=1.05)
        
        # Convert axes to array if there's only one client
        if num_clients == 1:
            axes = [axes]
            
        for client_id, ax in enumerate(axes):
            if client_id in distribution_info:
                info = distribution_info[client_id]
                sizes = [info['normal_samples'], info['attack_samples']]
                labels = ['Normal', info['attack_category']]
                ax.pie(sizes, labels=labels, autopct='%1.1f%%')
                ax.set_title(f'Client {client_id}')
        
        plt.tight_layout()
        plt.savefig("figures/all_clients_distribution.png")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting distribution: {e}")

def plot_global_metrics(history: pd.DataFrame):
    """Plot global metrics including loss, accuracy, and F1 vs AUC"""
    try:
        # Loss plot
        plt.figure(figsize=(10, 6))
        for metric in ['loss', 'val_loss']:
            if metric in history.columns:
                plt.plot(history[metric], label=metric.replace('_', ' ').title(), marker='o')
        plt.title('Global Loss vs Rounds')
        plt.xlabel('Rounds')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig("figures/global_loss.png")
        plt.close()
        
        # Combined metrics plot
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy']
        for metric in metrics:
            if metric in history.columns:
                plt.plot(history[metric], label=metric.replace('_', ' ').title(), marker='o')
        plt.title('Global Performance Metrics')
        plt.xlabel('Rounds')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig("figures/global_accuracy.png")
        plt.close()
    except Exception as e:
        print(f"Error in plot_global_metrics: {e}")

if __name__ == "__main__":
    # Define client_id with an initial value
    client_id = None

    df, num_cols, label_mapping, label_encoders = preprocess_data()
    client_data, _ = create_non_iid_data(df, num_cols)

    input_shape = len(num_cols)
    # print(f"Client {client_id} model input shape: {input_shape}")
    
    # Only perform tuning if hyperparameters don't exist
    # if client_id == 0 and not os.path.exists('best_hyperparameters.json'):
    #     # print(f"\nTuning hyperparameters (Client {client_id} is primary)...")
    #     best_hps = tune_hyperparameters()
    # else:
        # print(f"\nClient {client_id} loading existing hyperparameters...")
    best_hps = load_hyperparameters()
    
    # print(f"Client {client_id} using hyperparameters:", best_hps)

    model = create_model(input_shape=input_shape, best_hps=best_hps)

    X_val_combined = np.concatenate([client_data[i][2] for i in range(len(client_data))], axis=0)
    y_true = np.concatenate([client_data[i][3] for i in range(len(client_data))], axis=0)

    print(f"X_val_combined shape: {X_val_combined}")
    print(f"y_true shape: {y_true}")

    y_pred = model.predict(X_val_combined, verbose=0)
    y_pred_class = (y_pred >= 0.5).astype(int)
    print(f"y_pred shape: {y_pred}")
    y_score = y_pred.flatten()  # For binary classification

    history = pd.read_json("metrics_history.json")
    print("History DataFrame:", history.head())

    # Use binary classification
    classes = ['Normal', 'Attack']


    # Generate figures
    if client_id is None:
        analyze_results(history, y_true, y_pred_class, y_score, classes)
        # plot_distribution()
    else :
        # plot_metrics(history, client_id)    
        for client_id in range(4):
            analyze_results(history, y_true, y_pred, y_score, classes, client_id)


    plot_client_data_distribution()

