import tensorflow as tf
from tensorflow.keras import layers, models


def CNN_LSTM(seed):
    initializer = tf.keras.initializers.HeNormal(seed=seed)

    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(5, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Flatten(),
        layers.Reshape((-1, 15)),

        # layers.LSTM(units=50, return_sequences=True),
        layers.LSTM(units=50),
        
        layers.Flatten(),
        layers.Dense(10, activation='relu', kernel_initializer=initializer),
        layers.Dense(2)
    ], name = f'CNN_LSTM_{seed}')

    return model