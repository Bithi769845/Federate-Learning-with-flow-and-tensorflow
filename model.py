import tensorflow as tf
from hyperparameter_tuning import get_default_hyperparameters

def create_model(input_shape, best_hps=None):
    if best_hps is None:
        best_hps = get_default_hyperparameters()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            best_hps['units_1'], 
            activation='relu',
            input_shape=(input_shape,),
            kernel_regularizer=tf.keras.regularizers.l2(best_hps['l2_1'])
        ),
        tf.keras.layers.Dropout(best_hps['dropout_1']),
        tf.keras.layers.Dense(
            best_hps['units_2'], 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(best_hps['l2_2'])
        ),
        tf.keras.layers.Dropout(best_hps['dropout_2']),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps['lr']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
