# model.py
import tensorflow as tf
from tensorflow.keras import layers, models

# Defining a simple CNN model for brain tumor classification
model = models.Sequential([
    layers.Conv2D(256, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (tumorous/non-tumorous)
])

# To Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Displaying the model summary
model.summary()
