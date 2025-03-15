import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data():
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

    return df_combined, num_cols.tolist(), {'normal': 0, 'attack': 1}, label_encoders

def create_non_iid_data(df, num_cols, label_mapping, validation_split=0.2):
    client_data = {}
    distribution_info = {}

    attack_types = df['type'].unique().tolist()
    attack_types.remove('normal')

    # Split normal data per client
    normal_data = df[df['type'] == 'normal']
    num_clients = len(attack_types)
    normal_splits = np.array_split(normal_data, num_clients)

    for client_id, attack_type in enumerate(attack_types):
        normal_split = normal_splits[client_id]
        attack_data = df[df['type'] == attack_type]

        combined = pd.concat([normal_split, attack_data])
        distribution_info[client_id] = {
            'normal': len(normal_split),
            attack_type: len(attack_data)
        }

        X = combined[num_cols].values
        y = combined['label'].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        client_data[client_id] = (X_train, y_train, X_val, y_val)

    return client_data, distribution_info

if __name__ == "__main__":
    df, num_cols, label_mapping, label_encoders = preprocess_data()
    client_data, distribution = create_non_iid_data(df, num_cols, label_mapping)
    print("Data distribution per client:", distribution)