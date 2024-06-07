import tensorflow as tf
# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.load_model('./saved_models/RNN2/ECG_RNN_tf_8.h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

# Save the model.
with open('saved_models/ECG_RNN_tf_8.tflite', 'wb') as f:
  f.write(tflite_model)
