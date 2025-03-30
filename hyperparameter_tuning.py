import tensorflow as tf
from keras_tuner import RandomSearch
from data_preprocessing import preprocess_data, create_non_iid_data
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import os

def build_model(hp, input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        
        # First dense layer with tunable parameters
        tf.keras.layers.Dense(
            units=hp.Int('units_1', min_value=512, max_value=2048, step=256),
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(hp.Float('dropout_1', 0.2, 0.5)),
        
        # Second dense layer with tunable parameters
        tf.keras.layers.Dense(
            units=hp.Int('units_2', min_value=256, max_value=1024, step=128),
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(hp.Float('dropout_2', 0.1, 0.4)),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Tune learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_hyperparameters():
    print("\nStarting Hyperparameter Tuning...")
    
    # Load and preprocess data
    df, num_cols, label_mapping, _ = preprocess_data()
    input_shape = len(num_cols)
    print(f"Model input shape: {input_shape}")
    
    # Get client 1's data for tuning
    client_data, _ = create_non_iid_data(df, num_cols, label_mapping)
    X_train, y_train, X_val, y_val = client_data[1]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create tuner with model architecture
    tuner = RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        overwrite=True,
        directory='tuning_logs',
        project_name='fl_tuning'
    )
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Perform tuning
    print("\nSearching for best hyperparameters...")
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]
    
    # Evaluate best model
    test_loss, test_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
    
    # Save hyperparameters and connection info
    best_params = {
        'units_1': best_hp.get('units_1'),
        'units_2': best_hp.get('units_2'),
        'dropout_1': best_hp.get('dropout_1'),
        'dropout_2': best_hp.get('dropout_2'),
        'learning_rate': best_hp.get('learning_rate'),
        'tuning_accuracy': float(test_accuracy),
        'input_shape': input_shape  # Add input shape for verification
    }
    
    # Save both hyperparameters and tuning info
    save_hyperparameters(best_params)
    save_tuning_info(best_params, len(X_train), len(X_val))
    
    print("\nTuning Results:")
    print(f"Best validation accuracy: {test_accuracy:.4f}")
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"{param:15s}: {value}")
    
    print("\nBest hyperparameters saved to: best_hyperparameters.json")
    print("\nTuning information saved to: tuning_info.json")
    
    return best_params

def save_hyperparameters(best_hps):
    """Save hyperparameters to a file"""
    import json
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_hps, f)

def save_tuning_info(params, train_size, val_size):
    """Save tuning information for verification"""
    import json
    info = {
        'tuning_completed': True,
        'input_shape': params['input_shape'],
        'training_samples': train_size,
        'validation_samples': val_size,
        'tuning_accuracy': params['tuning_accuracy']
    }
    with open('tuning_info.json', 'w') as f:
        json.dump(info, f)

def load_hyperparameters():
    """Load hyperparameters from file or return defaults"""
    import json
    import os
    if os.path.exists('best_hyperparameters.json'):
        with open('best_hyperparameters.json', 'r') as f:
            return json.load(f)
    return get_default_hyperparameters()

def get_default_hyperparameters():
    # Simple, proven hyperparameters
    return {
        'units_1': 1024,
        'units_2': 512,
        'dropout_1': 0.3,
        'dropout_2': 0.2,
        'learning_rate': 0.001
    }

if __name__ == "__main__":
    try:
        best_params = tune_hyperparameters()
        print("\nTuning completed successfully!")
    except Exception as e:
        print(f"\nError during tuning: {e}")
