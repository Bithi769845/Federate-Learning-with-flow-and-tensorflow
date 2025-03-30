import tensorflow as tf
from hyperparameter_tuning import get_default_hyperparameters

def get_default_hyperparameters():
    return {
        'units_1': 1024,
        'units_2': 512,
        'dropout_1': 0.3,
        'dropout_2': 0.2,
        'learning_rate': 0.001
    }

def create_model(input_shape, best_hps=None):
    try:
        model = tf.keras.Sequential([
            # Input layer with strong normalization
            tf.keras.layers.InputLayer(input_shape=(input_shape,)),
            tf.keras.layers.BatchNormalization(),
            
            # Deep feature extraction
            tf.keras.layers.Dense(2048, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(1024, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.3),
            
            # Output layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc_roc')]
        )
        return model
    except Exception as e:
        print(f"Error in create_model: {e}")
        raise

