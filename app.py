from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
from enum import Enum

# Load your model
model = tf.keras.models.load_model('best_model.keras')


class Label(Enum):
    HEALTH = "Healthy"
    RUST = "Rust"
    OTHER = "Other"


app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file from the request
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Read the image using Pillow
        try:
            # Use stream to open the image directly
            image = Image.open(file.stream)
        except Exception as e:
            app.logger.error(f"Error reading JPEG image: {str(e)}")
            return jsonify({'error': 'Failed to read JPEG image'}), 400

        # Preprocess the image (resizing, normalizing, etc.)
        if image is not None:
            # Resize the image to (256, 256) and convert to numpy array
            image = image.resize((256, 256), Image.LANCZOS)
            image_array = np.array(image)

            # Ensure the image has three channels (RGB)
            if image_array.shape[-1] != 3:
                app.logger.error("Image does not have 3 channels (RGB).")
                return jsonify({'error': 'Image must be RGB'}), 400

            # Expand dimensions and normalize
            image_array = np.expand_dims(
                image_array, axis=0)  # Add batch dimension
            image_array = image_array / 255.0  # Normalize pixel values

            # Perform the prediction
            predictions = model.predict(image_array)

            # Convert the prediction probabilities to percentages
            prediction_percentages = predictions[0] * 100

            # Get the class with the highest score
            predicted_class = np.argmax(predictions, axis=-1)[0]

            # Convert the class index to a label
            if predicted_class == 1:
                predicted_label = Label.HEALTH.value
            elif predicted_class == 2:
                predicted_label = Label.RUST.value
            else:
                predicted_label = Label.OTHER.value

                # Create a response with both the predicted label and probabilities
            response = {
                'predicted_class': predicted_label,
                'probabilities': {
                    'Healthy': f"{prediction_percentages[1]:.2f}%",
                    'Rust': f"{prediction_percentages[2]:.2f}%",
                    'Other': f"{prediction_percentages[0]:.2f}%"
                }
            }

            # Return the prediction as JSON
            return jsonify(response)

        else:
            return jsonify({'error': 'Image could not be loaded'}), 400

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
