#This model.py

import tensorflow as tf

def create_model(input_size, hidden_size, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    return model
