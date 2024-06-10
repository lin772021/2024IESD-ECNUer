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


def CNNLSTM():
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        layers.Conv2D(filters=15, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Flatten(),  # 将卷积层输出扁平化处理
        layers.Reshape((-1, 35)),
        # layers.RNN(cell=SimpleRNNCell(units=32), return_sequences=True),
        # layers.RNN(cell=SimpleRNNCell(units=16), return_sequences=True),
        layers.LSTM(units=32, return_sequences=True),

        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        # layers.Dense(2),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

def CNN():
    initializer = tf.keras.initializers.HeNormal(seed=648)

    model = models.Sequential([
         # 增加卷积核的大小
        layers.Conv2D(filters=4, kernel_size=(10, 1), strides=(6, 1), padding='same', activation='relu', kernel_initializer=initializer),
        # 移除池化层，或者改为步幅更大的卷积层
        layers.Conv2D(filters=5, kernel_size=(10, 1), strides=(5, 1), padding='same', activation='relu', kernel_initializer=initializer),
        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(4, 1), padding='same', activation='relu', kernel_initializer=initializer),
        # 增大步幅
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(3, 1), padding='same', activation='relu', kernel_initializer=initializer),
        layers.Conv2D(filters=30, kernel_size=(2, 1), strides=(2, 1), padding='same', activation='relu', kernel_initializer=initializer),
        # layers.Flatten(),
        # layers.Reshape((-1, 25)),
        # layers.LSTM(units=32, return_sequences=True),
        # layers.LSTM(units=32, return_sequences=True),
        layers.Flatten(),
        layers.Dense(50, activation='relu', kernel_initializer=initializer),
        # layers.Dense(1, activation='sigmoid'),
        layers.Dense(2),
    ])

    return model


def CNN_AF():
    initializer = tf.keras.initializers.HeNormal(seed=821)

    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(10, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Conv2D(filters=5, kernel_size=(8, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Conv2D(filters=10, kernel_size=(6, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Conv2D(filters=10, kernel_size=(5, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Conv2D(filters=20, kernel_size=(5, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(200, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(100, activation='relu', kernel_initializer=initializer),
        layers.Dense(10, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(2),
        # layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)
    ])

    return model


def CNN1():
    initializer = tf.keras.initializers.HeNormal(seed=648)

    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(8, 1), strides=(2, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Conv2D(filters=5, kernel_size=(7, 1), strides=(2, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Conv2D(filters=10, kernel_size=(7, 1), strides=(2, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Conv2D(filters=20, kernel_size=(8, 1), strides=(2, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Conv2D(filters=20, kernel_size=(7, 1), strides=(2, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(200, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(10, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(2),
    ])

    return model


def CNN2():
    initializer = tf.keras.initializers.HeNormal(seed=648)

    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(16, 1), strides=(2, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Conv2D(filters=5, kernel_size=(15, 1), strides=(2, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Conv2D(filters=10, kernel_size=(15, 1), strides=(2, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Conv2D(filters=10, kernel_size=(15, 1), strides=(2, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Dropout(0.5),

        layers.Flatten(),
        # layers.Dense(100, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(10, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(2),
    ])

    return model

def CNN3():
    initializer = tf.keras.initializers.HeNormal(seed=648)

    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(85, 1), strides=(32, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(20, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        
        layers.Dropout(0.1),
        layers.Dense(10, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # layers.Dense(2),
        layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)
    ])

    return model
