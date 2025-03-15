import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List
import os
from data_preprocessing import create_non_iid_data, preprocess_data

def ensure_directory(directory="figures"):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_distribution():
    df, num_cols, label_mapping, label_encoders = preprocess_data()
    _, distribution = create_non_iid_data(df, num_cols, label_mapping)
    
    labels = ['Normal', 'DDoS', 'MITM', 'MQTT', 'Recon']
    client_data = [[dist.get(label.lower(), 0) for label in labels] for dist in distribution.values()]
    
    plt.figure(figsize=(8, 6))
    plt.pie(client_data[0], labels=labels, autopct='%1.1f%%')
    plt.title("Client Data Distribution")
    plt.savefig("figures/client_data_distribution.png")
    plt.close()

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
    if y_score.shape[1] != len(classes):
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 0])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    else:
        # Multi-class case
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{cls} (AUC = {auc_score:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/{filename}")
    plt.close()

def analyze_results(history: pd.DataFrame, y_true, y_pred, y_score, classes, client_id=None):
    if client_id is None:
        # Global federation results
        plot_metrics(history)
        plot_confusion_matrix(y_true, y_pred, classes, "confusion_global.png")
        plot_roc_curve(y_true, y_score, classes, "roc_global.png")
    else:
        # Skip client-specific confusion matrix and ROC curve
        plot_metrics(history, client_id)

def plot_client_data_distribution():
    df, num_cols, label_mapping, label_encoders = preprocess_data()
    _, distribution = create_non_iid_data(df, num_cols, label_mapping)
    
    ensure_directory()
    labels = ['Normal', 'DDoS', 'MITM', 'MQTT', 'Recon']
    fig, axes = plt.subplots(len(distribution), 1, figsize=(8, 6 * len(distribution)))
    
    if len(distribution) == 1:
        axes = [axes]
    
    for ax, (client_id, dist) in zip(axes, distribution.items()):
        client_data = [dist.get(label.lower(), 0) for label in labels]
        ax.pie(client_data, labels=labels, autopct='%1.1f%%')
        ax.set_title(f"Client {client_id} Data Distribution")
    
    plt.tight_layout()
    plt.savefig("figures/client_data_distribution.png")
    plt.close()

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

if __name__ == "__main__":
    # Example data for testing
    history = pd.DataFrame({
        'loss': np.random.rand(10),
        'val_loss': np.random.rand(10),
        'accuracy': np.random.rand(10),
        'val_accuracy': np.random.rand(10),
        'f1': np.random.rand(10),
        'auc_roc': np.random.rand(10)
    })
    y_true = np.random.randint(0, 5, 100)
    y_pred = np.random.randint(0, 5, 100)
    y_score = np.random.rand(100, 5)
    classes = ['Normal', 'DDoS', 'MITM', 'MQTT', 'Recon']
    client_accuracies = {0: 0.9, 1: 0.85, 2: 0.88, 3: 0.87}
    training_times = {0: 120, 1: 150, 2: 130, 3: 140}

    # Generate figures
    analyze_results(history, y_true, y_pred, y_score, classes)
    for client_id in range(4):
        analyze_results(history, y_true, y_pred, y_score, classes, client_id)
    plot_client_data_distribution()
    plot_client_accuracy_distribution(client_accuracies)
    plot_training_time_vs_clients(training_times)
