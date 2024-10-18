import os
from PIL import Image
import numpy as np
import tifffile as tiff

# Function to convert a .tif file to .jpeg with hyperspectral handling


def convert_tif_to_jpeg(tif_file, output_folder=None):
    try:
        # Try reading the .tif file using tifffile
        image = tiff.imread(tif_file)

        print(f"Loaded image with shape:"
              f"{image.shape} and dtype: {image.dtype}")

        # If the image has more than 3 channels, we'll use the first 3 channels as RGB
        if len(image.shape) == 3 and image.shape[-1] > 3:
            print(f"More than 3 channels found: "
                  f"{image.shape[-1]} channels. Using the first 3.")
            image = image[:, :, :3]  # Use the first 3 channels as RGB

        # If the image is single channel (grayscale or scientific data)
        elif len(image.shape) == 2:
            print(f"Single channel found. Converting to 3-channel grayscale.")
            # Convert grayscale to RGB by duplicating the channel
            image = np.stack([image] * 3, axis=-1)

        # Convert to uint8 (PIL requires 8-bit data)
        if image.dtype != np.uint8:
            image = (255 * (image / np.max(image))).astype(np.uint8)

        # Convert to a PIL image
        img = Image.fromarray(image)

    except Exception as e:
        print(f"Error reading TIFF: {str(e)}")
        return

    # Generate the output filename by changing the extension to .jpeg
    base_name = os.path.basename(tif_file)
    jpeg_name = os.path.splitext(base_name)[0] + ".jpeg"

    # Set the output path
    if output_folder:
        output_path = os.path.join(output_folder, jpeg_name)
    else:
        output_path = os.path.join(os.path.dirname(tif_file), jpeg_name)

    # Save the image as .jpeg
    img.save(output_path, "JPEG")
    print(f"Converted {tif_file} to {output_path}")

# Function to convert all .tif files in a directory to .jpeg


def convert_directory_tifs_to_jpegs(directory, output_folder=None):
    # Ensure the output folder exists
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif'):
                tif_file = os.path.join(root, file)
                convert_tif_to_jpeg(tif_file, output_folder)


# Example usage
# Specify the directory containing .tif files
input_directory = 'data/archive/train/Rust'
# Specify an output directory (optional)
output_directory = 'data/archive/train/Rust_jpeg'

convert_directory_tifs_to_jpegs(input_directory, output_directory)
