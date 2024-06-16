import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import RNN, SimpleRNNCell


def CNN_RNN1(seed):
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=15, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        
        layers.Flatten(),
        layers.Reshape((-1, 37)),
        layers.RNN(cell=SimpleRNNCell(units=64), return_sequences=True),

        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ], name = f'CNN_RNN1_{seed}')

    return model


def CNN_RNN2(seed):
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=15, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        
        layers.Flatten(),
        layers.Reshape((-1, 37)),
        layers.RNN(cell=SimpleRNNCell(units=32), return_sequences=True),
        layers.RNN(cell=SimpleRNNCell(units=16), return_sequences=True),

        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ], name = f'CNN_RNN2_{seed}')

    return model


def CNN_RNN3(seed):
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=15, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        
        layers.Flatten(),
        layers.Reshape((-1, 37)),
        layers.RNN(cell=SimpleRNNCell(units=32), return_sequences=True),
        layers.RNN(cell=SimpleRNNCell(units=16), return_sequences=True),
        layers.RNN(cell=SimpleRNNCell(units=10), return_sequences=True),

        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ], name = f'CNN_RNN3_{seed}')

    return model