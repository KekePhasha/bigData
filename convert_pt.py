import os
import torch
import numpy as np
import tensorflow as tf


def convert_pt_to_tf(input_directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through all the .pt files in the input directory
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.pt'):
                pt_file_path = os.path.join(root, file)

                # Load the PyTorch tensor from the .pt file
                tensor_data = torch.load(pt_file_path)

                # Convert the PyTorch tensor to a NumPy array
                numpy_data = tensor_data.numpy()

                # Convert the NumPy array to a TensorFlow tensor
                tf_tensor = tf.convert_to_tensor(numpy_data)

                # Save the TensorFlow tensor as a .npy file
                output_file_name = os.path.splitext(file)[0] + '.npy'
                output_file_path = os.path.join(
                    output_directory, output_file_name)

                # Save the TensorFlow tensor as a NumPy array in the output directory
                np.save(output_file_path, tf_tensor.numpy())
                print(f"Converted {file} to "
                      "output_file_name} and saved at {output_file_path}")


# Example usage
# Update with your dataset directory
# input_directory = 'data/ICPR01/kaggle/2'
# # Output directory in Kaggle
# output_directory = 'data/ICPR01/kaggle/traning/2'

# convert_pt_to_tf(input_directory, output_directory)
