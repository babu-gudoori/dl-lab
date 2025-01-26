import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np

# Step 1: Load and preprocess the MNIST dataset

# Load the MNIST dataset (train data only for autoencoder)
(x_train, _), (x_test, _) = mnist.load_data()

# Preprocess the data: Flatten the images and normalize them
x_train = x_train.reshape((x_train.shape[0], 784)) / 255.0
x_test = x_test.reshape((x_test.shape[0], 784)) / 255.0

# Define 'data' as the training set for the autoencoder
data = x_train  # You can use x_test as well if needed

# Step 2: Build the Autoencoder Model

# Define the input size and latent dimension
input_size = 784  # Input size for MNIST images (28x28 pixels flattened)
latent_dim = 32   # Latent space dimension (encoded representation)

# Encoder
input_data = Input(shape=(input_size,))
encoded = Dense(latent_dim, activation='relu')(input_data)

# Decoder
decoded = Dense(input_size, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_data, decoded)

# Step 3: Compile and Train the Autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder (data should be your input dataset)
autoencoder.fit(data, data, epochs=50, batch_size=32)

# Step 4: Encode Real-World Data (Optional)
# Create the encoder model to extract the encoded representation (latent space)
encoder = Model(input_data, encoded)

# Encode the data
encoded_data = encoder.predict(data)

# Step 5: Decode Encoded Data (Optional)
# Reconstruct the data from the encoded representation using the decoder
decoder_input = Input(shape=(latent_dim,))
decoded_output = autoencoder.layers[-1](decoder_input)  # Get the last layer of the decoder
decoder = Model(decoder_input, decoded_output)

# Reconstruct the data from encoded representation
reconstructed_data = decoder.predict(encoded_data)
