import tensorflow as tf

# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.load_model('./saved_models/AF/AFNet_acc_28.h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('saved_models/cnn_2.tflite', 'wb') as f:
  f.write(tflite_model)
