import os
import tensorflow as tf
import numpy as np
from PIL import Image  # Use Pillow for JPEG image processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Create an ImageDataGenerator with data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,  # Randomly rotate images by 40 degrees
    width_shift_range=0.2,  # Shift images horizontally by 20%
    height_shift_range=0.2,  # Shift images vertically by 20%
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zoom in
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Fill empty pixels with nearest pixel values
)


# Function to load and preprocess .jpeg images
def load_and_preprocess_image_jpeg(file_path, target_size=(256, 256)):
    # Open the JPEG image using Pillow
    img = Image.open(file_path)
    img = img.convert('RGB')  # Ensure the image is in RGB format

    # Resize to the target size
    img_resized = img.resize(target_size, Image.LANCZOS)

    # Convert to numpy array and normalize pixel values to [0, 1]
    rgb_image = np.array(img_resized) / 255.0

    return rgb_image


def augment_data(images):
    augmented_images = []

    # Loop through each image and apply augmentation
    for img in images:
        # Expand dimension to make it (1, 256, 256, 3)
        img = np.expand_dims(img, 0)
        for augmented_img in datagen.flow(img, batch_size=1):
            augmented_images.append(augmented_img[0])  # Append augmented image
            break  # Break after one augmentation per image

    return np.array(augmented_images)


# Function to load data from directories
def load_data(directory, target_size=(256, 256)):
    images = []
    labels = []

    # Loop through each class folder (e.g., 'Health', 'Rust', etc.)
    for class_folder in os.listdir(directory):
        class_folder_path = os.path.join(directory, class_folder)

        # Ensure the path is a directory
        if not os.path.isdir(class_folder_path):
            continue

        print(f"Processing folder: {class_folder}")  # Debugging statement

        # Loop through the files in the class folder
        for file_name in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, file_name)

            # Process only .jpeg files
            if file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.jpg'):
                try:
                    # Preprocess the image
                    image = load_and_preprocess_image_jpeg(
                        file_path, target_size)
                    images.append(image)
                    # Use the folder name as the label
                    labels.append(class_folder)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

# Apply data augmentation
    augmented_images = augment_data(images)

    # Combine original and augmented images
    images_combined = np.concatenate((images, augmented_images), axis=0)
    labels_combined = np.concatenate((labels, labels), axis=0)

    print(f"Total images loaded and augmented: {len(images_combined)}")
    print(f"Total labels loaded and augmented: {len(labels_combined)}")

    return images_combined, labels_combined


def load_single_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure the image is in RGB format

    # Resize the image to the target size
    img_resized = img.resize(target_size, Image.LANCZOS)

    # Convert to numpy array and normalize the image
    rgb_image = np.array(img_resized) / 255.0

    return rgb_image  # Return as numpy array
