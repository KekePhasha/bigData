from model import create_cnn_model
from data_prep import load_numpy_files
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from plotting import plot_training_history
import torch
import os


TRAIN_DIR = 'data/ICPR01/kaggle/traning/'

images, labels = load_numpy_files(TRAIN_DIR)

# Get data labels
unique_labels = np.unique(labels)

# Normalize the Data


def normalize_images(images):
    """Normalize pixel values to [0, 1]."""
    return images.astype('float32') / 255.0


images_normalized = normalize_images(images)

# Transpose the images to match the input shape of the model
images_normalized = np.transpose(images_normalized, (0, 2, 3, 1))


# Handle Class Labels
def encode_labels(labels):
    """Encode categorical labels to integers and convert to one-hot encoding."""
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    return to_categorical(integer_encoded)  # One-hot encode the integer labels


encoded_labels = encode_labels(labels)


# Check for empty arrays
if images_normalized.size == 0 or encoded_labels.size == 0:
    raise ValueError(
        "Images or labels are empty. Please check the data loading process.")


# Split the Dataset
def split_dataset(images, labels, test_size=0.2, random_state=42):
    """Split images and labels into training and testing sets."""
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)


X_train, X_test, y_train, y_test = split_dataset(
    images_normalized, encoded_labels)


# Example of using the data
print("Data preprocessing completed.")
print(f"Training data shape:"
      "{X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: "
      "{X_test.shape}, Testing labels shape: {y_test.shape}")


input_shape = (32, 32, 101)  # Image shape
num_classes = len(np.unique(labels))    # Number of classes

# Create a CNN model
model = create_cnn_model(input_shape, num_classes)

# Print the model summary
model.summary()

# Train the model
checkpoint = ModelCheckpoint(
    'best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Reduce learning rate if validation loss does not improve
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=1e-6)

# Stop training if validation loss does not improve after 5 epochs
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(X_test, y_test),
                    validation_split=0.2,   # Use 20% of the training data as validation data
                    callbacks=[checkpoint, reduce_lr, early_stopping])


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Plot the training history
plot_training_history(history)


def test_model_with_single_file(file_path):
    """Load a single .pt file, preprocess it, and predict using the model."""
    try:
        # Load the .pt file
        pt_tensor = torch.load(file_path)

        # Convert to NumPy array
        numpy_data = pt_tensor.numpy()

        # Normalize the new data
        numpy_data_normalized = normalize_images(numpy_data)
        numpy_data_normalized = np.transpose(
            numpy_data_normalized, (1, 2, 0))  # Shape to (32, 32, 101)

        # Expand dimensions to match input shape (1, height, width, channels)
        numpy_data_normalized = np.expand_dims(numpy_data_normalized, axis=0)

        # Make the prediction
        predictions = model.predict(numpy_data_normalized)

        # Convert the prediction probabilities to percentages
        prediction_percentages = predictions[0] * 100

        # Get the class with the highest score
        predicted_class = np.argmax(predictions, axis=-1)[0]

        # Prepare a response
        response = {
            'predicted_class': str(predicted_class),
            'probabilities': prediction_percentages.tolist()
        }

        print("Prediction results:", response)
        return response

    except Exception as e:
        print(f"Error testing model with file {file_path}: {str(e)}")


test_model_with_single_file(
    'data/ICPR01/kaggle/1/0503_0c8a97ab52d94546b35d81e7e830ffdb.pt')
