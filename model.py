from tensorflow.keras import layers, models, regularizers

# CNN Model


def create_cnn_model(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)

    # First Conv Block
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1,
                      padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Second Conv Block (with stride 2 for downsampling)
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Third Conv Block
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Fourth Conv Block (with stride 2 for downsampling)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Flattening the data
    x = layers.Flatten()(x)

    # Dense block
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.4)(x)  # Dropout to avoid overfitting

    # Output Layer with Softmax activation
    x = layers.Dense(num_classes)(x)
    output_layer = layers.Activation("softmax")(x)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
