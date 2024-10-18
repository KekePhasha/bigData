from model import create_cnn_model
from data_prep import load_data, load_single_image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
from plotting import plot_training_history


TRAIN_DIR = 'data/archive/train/jpeg/'

train_image, train_labels = load_data(TRAIN_DIR, target_size=(256, 256))

# Convert the labels to categorical (one-hot encoding) if they are in integer format
label_to_index = {label: idx for idx,
                  label in enumerate(np.unique(train_labels))}
train_labels = np.array([label_to_index[label] for label in train_labels])

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(
    train_labels, num_classes=len(label_to_index))

input_shape = (256, 256, 3)
num_classes = train_labels.shape[1]

print("Number of classes:", num_classes)

# Create a CNN model
model = create_cnn_model(input_shape, num_classes)

# Compile the model
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam', metrics=['accuracy'])

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
history = model.fit(train_image, train_labels,
                    epochs=10,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.2,   # Use 20% of the training data as validation data
                    callbacks=[checkpoint, reduce_lr, early_stopping])

# Save the model
model.save('final_model.keras')

# Plot the training history
plot_training_history(history)


# Load and preprocess a new image for prediction
# new_image = load_single_image(
#     'data/archive/val/val/101.tif', target_size=(256, 256))
# new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

# # Make a prediction
# predictions = model.predict(new_image)
# predicted_class = np.argmax(predictions, axis=-1)
# predicted_label = list(label_to_index.keys())[predicted_class[0]]

# print(f"Predicted class: {predicted_label}")
