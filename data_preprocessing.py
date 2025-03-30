import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

def preprocess_data():
    try:
        # Verify dataset directory exists
        if not os.path.exists("dataset"):
            raise FileNotFoundError("Dataset directory not found")
            
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

        # Load and combine datasets
        dataframes = []
        for file in attack_mapping.keys():
            df = pd.read_csv(f"dataset/{file}")
            df["Attack Name"] = file
            dataframes.append(df)
        
        # Load Benign Traffic separately
        benign_df = pd.read_csv("dataset/Benign Traffic.csv")
        benign_df["Attack Name"] = "Benign Traffic.csv"
        dataframes.append(benign_df)

        df_combined = pd.concat(dataframes, ignore_index=True)
        
        # Map specific attack names to categories
        df_combined['Attack Category'] = df_combined['Attack Name'].map(
            lambda x: attack_mapping.get(x, 'Benign')
        )
        
        # Data cleaning
        df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_combined.dropna(inplace=True)
        df_combined.drop_duplicates(inplace=True)

        # Create binary labels
        df_combined['Label'] = (df_combined['Attack Category'] != 'Benign').astype(int)

        # Process categorical and numerical features
        categorical_cols = df_combined.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['Attack Name', 'Label', 'Attack Category']]

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_combined[col] = le.fit_transform(df_combined[col].astype(str))
            label_encoders[col] = le

        num_cols = df_combined.select_dtypes(include=['float64', 'int64']).columns.tolist()
        num_cols = [col for col in num_cols if col not in ['Label']]

        scaler = StandardScaler()
        df_combined[num_cols] = scaler.fit_transform(df_combined[num_cols])

        print("\nProcessed data summary:")
        print(f"Total samples: {len(df_combined)}")
        print(f"Normal samples: {len(df_combined[df_combined['Label'] == 0])}")
        print(f"Attack samples: {len(df_combined[df_combined['Label'] == 1])}")

        return df_combined, num_cols, label_encoders, attack_mapping

    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        raise

def create_non_iid_data(df, num_cols, attack_mapping, validation_split=0.2):
    client_data = {}
    distribution_info = {}

    # Separate normal and attack data
    normal_data = df[df['Label'] == 0]
    attack_data = df[df['Label'] == 1]

    # Get unique attack categories and their sample counts
    attack_counts = attack_data['Attack Category'].value_counts()
    attack_categories = attack_counts.index.tolist()
    num_clients = len(attack_categories)

    # Calculate normal data allocation ratios based on attack proportions
    total_attacks = attack_counts.sum()
    normal_ratios = attack_counts / total_attacks

    # Create imbalanced normal data splits
    normal_subsets = []
    remaining_normal = normal_data.copy()
    
    for i in range(num_clients - 1):
        # Calculate size based on attack proportion
        subset_size = int(len(normal_data) * normal_ratios.iloc[i])
        subset = remaining_normal.sample(n=subset_size, random_state=42+i)
        normal_subsets.append(subset)
        remaining_normal = remaining_normal.drop(subset.index)
    normal_subsets.append(remaining_normal)

    # Assign data to clients
    for client_id, (normal_subset, attack_category) in enumerate(zip(normal_subsets, attack_categories), 1):
        attack_subset = attack_data[attack_data['Attack Category'] == attack_category]
        combined = pd.concat([normal_subset, attack_subset])
        
        distribution_info[client_id] = {
            'normal_samples': len(normal_subset),
            'attack_samples': len(attack_subset),
            'attack_category': attack_category
        }

        X_train, X_val, y_train, y_val = train_test_split(
            combined[num_cols],
            combined['Label'],
            test_size=validation_split,
            stratify=combined['Label'],
            random_state=42
        )
        client_data[client_id] = (X_train, y_train, X_val, y_val)

        print(f"\nClient {client_id}:")
        print(f"- Attack category: {attack_category}")
        print(f"- Normal samples: {len(normal_subset)}")
        print(f"- Attack samples: {len(attack_subset)}")
        print(f"- Total samples: {len(combined)}")

    with open("distribution_info.pkl", "wb") as f:
        pickle.dump(distribution_info, f)

    return client_data, distribution_info

if __name__ == "__main__":
    df, num_cols, label_encoders, attack_mapping = preprocess_data()
    client_data, distribution = create_non_iid_data(df, num_cols, attack_mapping)
    print("Data distribution per client:", distribution)