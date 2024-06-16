import tensorflow as tf

def prepare_trained_model(trained_model):
    """Fix the input of the trained model for inference

        Args:
            trained_model (tf.keras.Model): the trained LSTM model

        Returns:
            run_model (tf.keras.Model): the trained model with fixed input tensor size for inference
    """
    # TFLite converter requires fixed shape input to work, alternative: b/225231544
    fixed_input = tf.keras.layers.Input(shape=[1250, 1, 1],
                                        batch_size=1,
                                        dtype=trained_model.inputs[0].dtype,
                                        name="fixed_input")
    fixed_output = trained_model(fixed_input)
    run_model = tf.keras.models.Model(fixed_input, fixed_output)
    return run_model

# Create a model using high-level tf.keras.* APIs
trained_model = tf.keras.models.load_model('./saved_models/LSTM_seed_all/LSTM_seed_all_acc_14.h5')
run_model = prepare_trained_model(trained_model)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(run_model)
tflite_model = converter.convert()

# Save the model.
with open('saved_models/LSTM.tflite', 'wb') as f:
    f.write(tflite_model)