import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import RNN, SimpleRNNCell


def AFNet():
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=15, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Flatten(),  # 将卷积层输出扁平化处理，以便输入到全连接层
        # layers.Dropout(0.5),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model

def RNN1():
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=15, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        
        layers.Flatten(),  # 将卷积层输出扁平化处理
        layers.Reshape((-1, 37)),
        layers.RNN(cell=SimpleRNNCell(units=64), return_sequences=True),

        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model

# current best
def RNN2():
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=15, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        
        layers.Flatten(),  # 将卷积层输出扁平化处理
        layers.Reshape((-1, 37)),
        layers.RNN(cell=SimpleRNNCell(units=32), return_sequences=True),
        layers.RNN(cell=SimpleRNNCell(units=16), return_sequences=True),

        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model

def RNN3():
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=15, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        
        layers.Flatten(),  # 将卷积层输出扁平化处理
        layers.Reshape((-1, 37)),
        layers.RNN(cell=SimpleRNNCell(units=32), return_sequences=True),
        layers.RNN(cell=SimpleRNNCell(units=16), return_sequences=True),
        layers.RNN(cell=SimpleRNNCell(units=10), return_sequences=True),

        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model

def RNN():
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=15, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        
        layers.Flatten(),  # 将卷积层输出扁平化处理
        layers.Reshape((-1, 17)),
        # layers.RNN(cell=SimpleRNNCell(units=32), return_sequences=True),
        # layers.RNN(cell=SimpleRNNCell(units=16), return_sequences=True),
        layers.LSTM(units=32, return_sequences=False),

        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model