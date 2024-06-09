import tensorflow as tf
from tensorflow.keras import layers, models


def AFNet():
    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(6, 1), strides=(1, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(1, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(1, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(1, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(1, 1), padding='valid', activation='relu'),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        layers.Dropout(0.5),
        layers.Flatten(),  # 将卷积层输出扁平化处理，以便输入到全连接层
        # layers.Dropout(0.5),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model

def CNN_AF():
    initializer = tf.keras.initializers.HeNormal(seed=648)

    model = models.Sequential([
        layers.Conv2D(filters=3, kernel_size=(5, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),
        
        layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=initializer),
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
        layers.Dense(100, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(10, activation='relu', kernel_initializer=initializer), #, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(2),
    ])

    return model