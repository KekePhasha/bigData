from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import tensorflow as tf
# ENUM
from enum import Enum

# Define an Enum for the class labels


class PredictionLabel(Enum):
    MILD = 0
    SERIOUS = 1


# Load your model
model = tf.keras.models.load_model('best_model.keras')

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file from the request (assuming it's a .pt file)
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Load the .pt file and convert to NumPy array
        try:
            pt_tensor = torch.load(file)
            numpy_data = pt_tensor.numpy()

            # Debug: Print the shape of the tensor
            # Should be (101, 32, 32)
            print(f"Loaded tensor shape: {numpy_data.shape}")

            # Ensure the shape is as expected
            if len(numpy_data.shape) == 3 and numpy_data.shape[0] == 101:
                # Transpose to (height, width, channels) and add batch dimension
                numpy_data = np.transpose(
                    numpy_data, (1, 2, 0))  # Shape: (32, 32, 101)
                # Shape: (1, 32, 32, 101)
                numpy_data = np.expand_dims(numpy_data, axis=0)
            else:
                return jsonify({'error': f"Unexpected tensor shape: {numpy_data.shape}"}), 400

            # Normalize the data if necessary
            # Assuming normalization is required
            numpy_data = numpy_data.astype('float32') / 255.0

            # Debug: Print the shape of the tensor after processing
            # Should be (1, 32, 32, 101)
            print(f"Processed tensor shape: {numpy_data.shape}")

            # Perform the prediction
            predictions = model.predict(numpy_data)

            # Debug: Print raw predictions
            print(f"Raw predictions: {predictions}")

            # Convert the prediction probabilities to percentages
            prediction_percentages = predictions[0] * 100

            # Get the class with the highest score
            predicted_class = np.argmax(predictions, axis=-1)[0]

            # Convert the class index to the corresponding label
            predicted_class = PredictionLabel(predicted_class).name

            # Prepare a response
            response = {
                'predicted_class': predicted_class,
                'prediction_percentages': {
                    PredictionLabel.MILD.name: float(prediction_percentages[PredictionLabel.MILD.value]),
                    PredictionLabel.SERIOUS.name: float(
                        prediction_percentages[PredictionLabel.SERIOUS.value])
                }
            }

            print(response)

            # Return the prediction as JSON
            return jsonify(response)

        except Exception as e:
            app.logger.error(f"Error processing .pt file: {str(e)}")
            return jsonify({'error': 'Failed to process .pt file'}), 400

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
