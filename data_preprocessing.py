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
        # print(f"Data loaded from {file}, shape: {df.shape}")
        df['type'] = attack_type
        dataframes.append(df)
        # print(f"Total dataframes loaded: {dataframes}")
    
    df_combined = pd.concat(dataframes, ignore_index=True)
    # print("Unique attack types in dataset:", df_combined['type'].unique())

    # Data cleaning
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_combined.dropna(inplace=True)
    df_combined.drop_duplicates(inplace=True)
    # print(f"Combined dataset shape: {df_combined.shape}")

    # print(f"Unique attack types in dataset: {df_combined['type'].unique}")
    # Label encoding
    # df_combined['label'] = df_combined['type'].map(lambda x: 0 if x == 'normal' else 1)
    # Print the unique labels and their counts
    # label_counts = df_combined['label'].value_counts()
    # print(f"Label encoding completed. Unique labels: {df_combined['label'].unique()}")
    # print(f"Count of 0: {label_counts.get(0, 0)}")
    # print(f"Count of 1: {label_counts.get(1, 0)}") 
    # print(f"Combined dataset shape after cleaning: {df_combined.shape}")
    # df_combined.to_csv('df_combined.csv', index=False)
    categorical_cols = df_combined.select_dtypes(include=['object']).columns.tolist()
    # print(f"Categorical columns before encoding: {categorical_cols}")
    categorical_cols.remove('type')
    # print(f"Categorical columns before encoding: {categorical_cols}")
    # df_combined.to_csv('df_combined.csv', index=False)
    # print(f"Categorical columns after removing 'type': {categorical_cols}")
    # df_combined['label'] = df_combined['type'].map(attack_mapping)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_combined[col] = le.fit_transform(df_combined[col])
        # print(f"Encoded {col} with classes: {le.classes_}")
        label_encoders[col] = le
        # print(f"Label encoding completed for {col}. Unique values: {df_combined[col].unique()}")

    # print("label encoders", label_encoders)
    # print(df_combined.dtypes)
    # Feature scaling
    num_cols = df_combined.select_dtypes(include=['float64', 'int64', 'int32']).columns.drop('Label', errors='ignore')
    # print(f"Numerical columns before scaling: {num_cols.shape}")
    # num_cols.to_csv('num_cols.csv', index=False)
    scaler = StandardScaler()
    df_combined[num_cols] = scaler.fit_transform(df_combined[num_cols])

    # df_combined.to_csv('df_combined.csv', index=False)
    # print("Combined total shape ",df_combined.shape)
    # print("Number of Columns", num_cols.shape)
    # print("Attack Mapping ", attack_mapping)
    return df_combined, num_cols.tolist(), label_encoders, attack_mapping

def create_non_iid_data(df, num_cols, validation_split=0.2):
    client_data = {}
    distribution_info = {}

    # Separate normal and attack data
    normal_data = df[df['type'] == 'normal']
    attack_data = df[df['type'] != 'normal']

    # print(f"Normal data shape: {normal_data.shape}")
    # print(f"Attack data shape: {attack_data}")
    # Get unique attack categories
    attack_categories = sorted(attack_data['type'].unique())

    # print(f"Unique attack categories: {attack_categories}")

    num_clients = len(attack_categories)

    # print(f"Number of clients: {num_clients}")

    # Split normal data proportionally
    normal_splits = np.array_split(normal_data, num_clients)

    # print(normal_splits)
    # print(f"Normal data split into {len(normal_splits)} clients.")
    # print(f"Normal data split sizes: {[len(split) for split in normal_splits]}")

    # Add verification prints
    # print(f"\nCreating Non-IID data distribution:")
    # print(f"Found {num_clients} unique attack categories: {attack_categories}")

    # Create client datasets (start from 0 instead of 1)
    for client_id, (normal_split, attack_category) in enumerate(zip(normal_splits, attack_categories)):
        attack_subset = attack_data[attack_data['type'] == attack_category]
        # normal_split.to_csv(f"normal_split_{client_id}.csv", index=False)
        # print(f"Normal split {client_id} shape: {normal_split.shape}")
        # print("Attact subset shape", attack_subset.shape)
        # print(f"Client {client_id}: Attack category: {attack_category}, Normal samples: {len(normal_split)}, Attack samples: {len(attack_subset)}")
        combined = pd.concat([normal_split, attack_subset])

        # combined.to_csv(f"client_{client_id}_data.csv", index=False)
        # print(f"Combined data shape for client {client_id}: {combined.shape}")
        
        distribution_info[client_id] = {
            'normal_samples': len(normal_split),
            'attack_samples': len(attack_subset),
            'attack_category': attack_category
        }

        # Add verification prints
        # print(f"\nClient {client_id}:")
        # print(f"- Attack category: {attack_category}")
        # print(f"- Normal samples: {len(normal_split)}")
        # print(f"- Attack samples: {len(attack_subset)}")
        # print(f"- Total samples: {len(combined)}")

        # print("number of column", num_cols)
        X = combined[num_cols].values
        # print(f"X shape {X}")
        y = combined['Label'].values
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        # print(f"X_train shape {X_train.shape} | y_train shape {y_train.shape} | X_val shape {X_val.shape} | y_val shape {y_val.shape} | y_val shape {y_val.shape}") 

        client_data[client_id] = (X_train, y_train, X_val, y_val)
        # print("Client Data ",client_data[client_id])
    # Save distribution info for visualization
    with open("distribution_info.pkl", "wb") as f:
        pickle.dump(distribution_info, f)
            # Debug: print label distributions for each client
    # for client_id, (X_train, y_train, X_val, y_val) in client_data.items():
    #     print(f"[Client {client_id}] Unique Train Labels: {np.unique(y_train)} | Unique Val Labels: {np.unique(y_val)}")


    return client_data, distribution_info

if __name__ == "__main__":
    df, num_cols, label_encoders, attack_mapping = preprocess_data()
    # print("Whole data shape", df.shape)
    client_data, distribution = create_non_iid_data(df, num_cols)
    print("Data distribution per client:", distribution)