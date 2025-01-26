import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
train_data = np.random.rand(100, 20)
train_labels = np.random.randint(2, size=(100,))

# Create a simple neural network
model = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(20,)),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=5, batch_size=10)
