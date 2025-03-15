import tensorflow as tf
from keras_tuner import RandomSearch
from data_preprocessing import preprocess_data, create_non_iid_data
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def build_model(hp, input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(
            units=hp.Int('units_1', 64, 512, step=64),
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float('l2_1', 1e-4, 1e-2, sampling='log'))
        ),
        tf.keras.layers.Dropout(hp.Float('dropout_1', 0.2, 0.5)),
        tf.keras.layers.Dense(
            units=hp.Int('units_2', 32, 256, step=32),
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float('l2_2', 1e-4, 1e-2, sampling='log'))
        ),
        tf.keras.layers.Dropout(hp.Float('dropout_2', 0.2, 0.5)),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output layer
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log'))
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def save_hyperparameters(best_hps):
    """Save hyperparameters to a file"""
    import json
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_hps, f)

def load_hyperparameters():
    """Load hyperparameters from file or return defaults"""
    import json
    import os
    if os.path.exists('best_hyperparameters.json'):
        with open('best_hyperparameters.json', 'r') as f:
            return json.load(f)
    return get_default_hyperparameters()

def get_default_hyperparameters():
    return {
        'units_1': 256,
        'l2_1': 1e-3,
        'dropout_1': 0.3,
        'units_2': 128,
        'l2_2': 1e-3,
        'dropout_2': 0.3,
        'lr': 1e-3
    }

def tune_hyperparameters():
    # Load and preprocess data
    df, num_cols, label_mapping, label_encoders = preprocess_data()
    input_shape = len(num_cols)
    print(f"Tuning model input shape: {input_shape}")
    
    # Create non-IID data
    client_data, _ = create_non_iid_data(df, num_cols, label_mapping)
    X_train, y_train, X_val, y_val = client_data[0]
    
    tuner = RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=10,  # Increased from 5
        executions_per_trial=3,  # Increased from 2
        directory='tuning',
        project_name='fl_tuning'
    )
    
    print("Starting hyperparameter tuning...")
    tuner.search(
        X_train, y_train,
        epochs=15,  # Increased from 10
        batch_size=32,  # Added batch_size
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]  # Added early stopping
    )
    
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("\nBest hyperparameters found:", best_hps.values)
    
    # Convert to dictionary format and save
    best_params = {
        'units_1': best_hps.get('units_1'),
        'l2_1': best_hps.get('l2_1'),
        'dropout_1': best_hps.get('dropout_1'),
        'units_2': best_hps.get('units_2'),
        'l2_2': best_hps.get('l2_2'),
        'dropout_2': best_hps.get('dropout_2'),
        'lr': best_hps.get('lr')
    }
    save_hyperparameters(best_params)
    return best_params

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print("Best hyperparameters:", best_params.values)
