import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def preprocess_data():
    # Attack labels mapping
    attack_mapping = {
        'DDoS ICMP Flood.csv': 'DoS',
        'DDoS UDP Flood.csv': 'DoS',
        'DoS ICMP Flood.csv': 'DoS',
        'DoS UDP Flood.csv': 'DoS',
        'MITM ARP Spoofing.csv': 'MITM',
        'MQTT DoS Publish Flood.csv': 'MQTT',
        'MQTT Malformed.csv': 'MQTT',
        'Recon Ping Sweep.csv': 'Recon',
        'Recon Vulnerability Scan.csv': 'Recon'
    }

    attack_labels = {
        "Benign Traffic.csv": "normal",
        "DDoS ICMP Flood.csv": "DoS",
        "DDoS UDP Flood.csv": "DoS",
        "DoS ICMP Flood.csv": "DoS",
        "DoS UDP Flood.csv": "DoS",
        "MITM ARP Spoofing.csv": "MITM",
        "MQTT DoS Publish Flood.csv": "MQTT",
        "MQTT Malformed.csv": "MQTT",
        "Recon Ping Sweep.csv": "Recon",
        "Recon Vulnerability Scan.csv": "Recon"
    }

    dataframes = []
    for file, attack_type in attack_labels.items():
        df = pd.read_csv(f"dataset/{file}")
        print(f"Data loaded from {file}, shape: {df.shape}")
        df['type'] = attack_type
        dataframes.append(df)

    df_combined = pd.concat(dataframes, ignore_index=True)
    print("Unique attack types in dataset:", df_combined['type'].unique())

    # Data cleaning
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_combined.dropna(inplace=True)
    df_combined.drop_duplicates(inplace=True)

    # Label encoding
    df_combined['label'] = df_combined['type'].map(lambda x: 0 if x == 'normal' else 1)

    categorical_cols = df_combined.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('type')

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col])
        label_encoders[col] = le

    # Feature scaling
    num_cols = df_combined.select_dtypes(include=['float64', 'int64']).columns.drop('label', errors='ignore')
    scaler = StandardScaler()
    df_combined[num_cols] = scaler.fit_transform(df_combined[num_cols])

    return df_combined, num_cols.tolist(), label_encoders, attack_mapping

def create_non_iid_data(df, num_cols, attack_mapping, validation_split=0.2):
    client_data = {}
    distribution_info = {}

    # Separate normal and attack data
    normal_data = df[df['type'] == 'normal']
    attack_data = df[df['type'] != 'normal']

    # Get unique attack categories
    attack_categories = sorted(attack_data['type'].unique())
    num_clients = len(attack_categories)

    # Split normal data proportionally
    normal_splits = np.array_split(normal_data, num_clients)

    # Add verification prints
    print(f"\nCreating Non-IID data distribution:")
    print(f"Found {num_clients} unique attack categories: {attack_categories}")

    # Create client datasets (start from 0 instead of 1)
    for client_id, (normal_split, attack_category) in enumerate(zip(normal_splits, attack_categories)):
        attack_subset = attack_data[attack_data['type'] == attack_category]
        combined = pd.concat([normal_split, attack_subset])
        
        distribution_info[client_id] = {
            'normal_samples': len(normal_split),
            'attack_samples': len(attack_subset),
            'attack_category': attack_category
        }

        # Add verification prints
        print(f"\nClient {client_id}:")
        print(f"- Attack category: {attack_category}")
        print(f"- Normal samples: {len(normal_split)}")
        print(f"- Attack samples: {len(attack_subset)}")
        print(f"- Total samples: {len(combined)}")

        X = combined[num_cols].values
        y = combined['label'].values
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        client_data[client_id] = (X_train, y_train, X_val, y_val)

    # Save distribution info for visualization
    with open("distribution_info.pkl", "wb") as f:
        pickle.dump(distribution_info, f)
            # Debug: print label distributions for each client
    for client_id, (X_train, y_train, X_val, y_val) in client_data.items():
        print(f"[Client {client_id}] Unique Train Labels: {np.unique(y_train)} | Unique Val Labels: {np.unique(y_val)}")


    return client_data, distribution_info

if __name__ == "__main__":
    df, num_cols, label_encoders, attack_mapping = preprocess_data()
    client_data, distribution = create_non_iid_data(df, num_cols, attack_mapping)
    print("Data distribution per client:", distribution)