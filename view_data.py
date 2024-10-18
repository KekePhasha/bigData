import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Open the .tif file using rasterio
file_path = 'data/archive/train/Health/hyper (1).tif'

with rasterio.open(file_path) as src:
    # Read the red, green, and blue bands (adjust band numbers based on your data)
    red = src.read(3)   # Assuming band 3 is the red channel
    green = src.read(2)  # Assuming band 2 is the green channel
    blue = src.read(1)  # Assuming band 1 is the blue channel

    # Stack the bands to create an RGB image
    rgb = np.dstack((red, green, blue))

    # Normalize the values for better display
    rgb = rgb / np.max(rgb)

    # Plot the RGB image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title('RGB Composite Image')
    plt.show()
