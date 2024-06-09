import tensorflow as tf

  

if __name__ == '__main__':
  # print('tf version: {}'.format(tf.__version__))

  # print('keras version: {}'.format(tf.keras.__version__))

  # print('GPU: {}'.format(tf.test.is_gpu_available())) # 1.x版本的TF使用此行

  print('GPU Nums: {}'.format(len(tf.config.list_physical_devices('GPU')))) # 2.x版本的TF使用