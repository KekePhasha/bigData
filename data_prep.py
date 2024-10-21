import os
import numpy as np


# Function to load NumPy files from a directory and use subfolder names as labels
def load_numpy_files(directory):
    images = []
    labels = []

    for class_folder in os.listdir(directory):
        class_folder_path = os.path.join(directory, class_folder)

        # Ensure the path is a directory
        if not os.path.isdir(class_folder_path):
            continue

        for file_name in os.listdir(class_folder_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_folder_path, file_name)
                # Load the NumPy array
                image_data = np.load(file_path)
                images.append(image_data)

                # Use the subfolder name as the label
                labels.append(class_folder)

    if images:
        return np.array(images), np.array(labels)
    else:
        raise ValueError("No images found in the directory.")
