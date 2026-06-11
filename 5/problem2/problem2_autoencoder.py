import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
max_features = 10000
maxlen = 200
encoding_dim = 32  # Starting with 32 dimensions for the encoded representation

# Load and prepare the data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# Build the autoencoder
input_review = Input(shape=(maxlen,))
# Encoder
encoded = Dense(128, activation='relu')(input_review)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(maxlen, activation='sigmoid')(decoded)

autoencoder = Model(input_review, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
history = autoencoder.fit(X_train, X_train,
                          epochs=20,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(X_test, X_test))

# Plot the loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Loss Curves for Autoencoder with {encoding_dim} Codings')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./5/problem2/problem2_loss_curves_{encoding_dim}.png')
plt.close()

# Decode reviews back to text
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def decode_review(encoded_review):
    # The +3 offset is because 0 is for padding, 1 is for start of sequence, and 2 is for unknown
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Select 5 random samples and visualize them
decoded_reviews = autoencoder.predict(X_test)

# Since the output of the decoder is float values between 0 and 1, 
# we need to scale and round them to get back integer word indices.
# A simple approach is to scale by max_features and round.
decoded_reviews_int = np.round(decoded_reviews * max_features).astype(int)


random_indices = np.random.choice(X_test.shape[0], 5, replace=False)

for i in random_indices:
    print(f"--- Sample {i} ---")
    print("Original Review:")
    print(decode_review(X_test[i]))
    print("\nReconstructed Review:")
    print(decode_review(decoded_reviews_int[i]))
    print("\n")