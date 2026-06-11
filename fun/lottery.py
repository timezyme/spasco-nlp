# Keras Encoder-Decoder Model for Lottery Number Sequence Prediction
# This is a fun, educational exercise to demonstrate a sequence-to-sequence architecture.
# It is not intended to be a real lottery prediction tool.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# --- 1. Configuration & Hyperparameters ---
# These are the main settings you can tweak for your experiment.

# Data parameters
NUM_SAMPLES = 10000       # How many historical draws to simulate.
SEQUENCE_LENGTH = 6       # The number of balls in a lottery draw (e.g., 6).
MAX_LOTTERY_NUMBER = 50   # The highest possible number in the lottery (e.g., 1 to 50).

# Model parameters
LATENT_DIM = 256          # The size of the "thought vector" (encoder's memory).
BATCH_SIZE = 64           # Number of samples to process at once during training.
EPOCHS = 20               # How many times to go through the entire dataset during training.


# --- 2. Data Simulation & Preparation ---
# We'll generate fake lottery data for this example.
# In a real scenario, you would load historical lottery results here.

print("Generating simulated lottery data...")
# We use sets to ensure no duplicate numbers in a single draw
historical_draws = [sorted(list(np.random.choice(range(1, MAX_LOTTERY_NUMBER + 1), SEQUENCE_LENGTH, replace=False))) for _ in range(NUM_SAMPLES)]

# We need to define our "vocabulary" - all possible numbers + special tokens
# 0 is reserved for padding/masking.
# MAX_LOTTERY_NUMBER + 1 will be our "start of sequence" token.
# MAX_LOTTERY_NUMBER + 2 will be our "end of sequence" token.
num_tokens = MAX_LOTTERY_NUMBER + 3
token_index = {i: i for i in range(num_tokens)}
reverse_token_index = {i: i for i in range(num_tokens)}
start_token = MAX_LOTTERY_NUMBER + 1
end_token = MAX_LOTTERY_NUMBER + 2

# The encoder input is the historical sequence.
encoder_input_data_raw = np.array(historical_draws)

# The decoder input starts with a "start" token and then has the sequence.
decoder_input_data_raw = np.zeros_like(encoder_input_data_raw)
decoder_input_data_raw[:, 0] = start_token
decoder_input_data_raw[:, 1:] = encoder_input_data_raw[:, :-1]

# The decoder target is the sequence shifted one step, ending with an "end" token.
decoder_target_data_raw = np.zeros_like(encoder_input_data_raw)
decoder_target_data_raw[:, :-1] = encoder_input_data_raw[:, 1:]
decoder_target_data_raw[:, -1] = end_token

# --- 3. One-Hot Encode the Data ---
# Neural networks work with vectors, not raw numbers. We convert each number
# into a vector of zeros with a '1' at the index corresponding to the number.

print("Vectorizing data (one-hot encoding)...")
encoder_input_data = np.zeros((NUM_SAMPLES, SEQUENCE_LENGTH, num_tokens), dtype="float32")
decoder_input_data = np.zeros((NUM_SAMPLES, SEQUENCE_LENGTH, num_tokens), dtype="float32")
decoder_target_data = np.zeros((NUM_SAMPLES, SEQUENCE_LENGTH, num_tokens), dtype="float32")

for i, seq in enumerate(encoder_input_data_raw):
    for t, num in enumerate(seq):
        encoder_input_data[i, t, num] = 1.0

for i, seq in enumerate(decoder_input_data_raw):
    for t, num in enumerate(seq):
        decoder_input_data[i, t, num] = 1.0

for i, seq in enumerate(decoder_target_data_raw):
    for t, num in enumerate(seq):
        decoder_target_data[i, t, num] = 1.0


# --- 4. Build the Encoder-Decoder Model ---

# === Encoder ===
# The encoder reads the input sequence and compresses it into a state vector (thought vector).
encoder_inputs = Input(shape=(None, num_tokens), name="encoder_input")
encoder_lstm = LSTM(LATENT_DIM, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm(encoder_inputs)
# We discard the encoder outputs and only keep the final states (h and c).
encoder_states = [state_h, state_c]

# === Decoder ===
# The decoder takes the encoder's state vector and generates the output sequence.
decoder_inputs = Input(shape=(None, num_tokens), name="decoder_input")
# We set up the decoder to return sequences and to use the encoder's states as its initial state.
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# A dense layer to predict the probability of each token (number) at each time step.
decoder_dense = Dense(num_tokens, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# === The Full Model ===
# This model turns `encoder_input_data` & `decoder_input_data` into `decoder_target_data`.
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("\n--- Training Model Summary ---")
model.summary()

# --- 5. Train the Model ---
print("\nStarting training...")
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2, # Use 20% of the data for validation
)

# --- 6. Build Inference Models ---
# After training, we need separate models for prediction (inference) because
# we will generate the output one step at a time.

# Encoder model: takes a sequence, returns the "thought vector".
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model: takes the "thought vector" and the previous predicted number
# to predict the next number in the sequence.
decoder_state_input_h = Input(shape=(LATENT_DIM,), name="decoder_state_h")
decoder_state_input_c = Input(shape=(LATENT_DIM,), name="decoder_state_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

print("\n--- Inference Encoder Model Summary ---")
encoder_model.summary()
print("\n--- Inference Decoder Model Summary ---")
decoder_model.summary()


# --- 7. Prediction Function ---
def decode_sequence(input_seq):
    # Encode the input sequence to get the initial decoder state.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate an empty target sequence of length 1.
    # Populate the first token of the target sequence with the start token.
    target_seq = np.zeros((1, 1, num_tokens))
    target_seq[0, 0, start_token] = 1.0

    stop_condition = False
    decoded_sequence = []
    
    while not stop_condition:
        # Predict the next token
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample a token (we take the one with the highest probability)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # Stop if we hit the end token or max length
        if sampled_token_index == end_token or len(decoded_sequence) >= SEQUENCE_LENGTH:
            stop_condition = True
        else:
            # Add the sampled token to our sequence
            if sampled_token_index > 0: # Avoid adding the padding token
                 decoded_sequence.append(sampled_token_index)


        # Update the target sequence for the next prediction
        target_seq = np.zeros((1, 1, num_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update the states
        states_value = [h, c]
        
    return decoded_sequence

# --- 8. Run an Example Prediction ---
print("\n--- Making a Prediction ---")
# Pick a random sequence from our data to use as a prompt
i = np.random.randint(0, NUM_SAMPLES)
input_seq_raw = encoder_input_data_raw[i : i + 1][0]
input_seq_onehot = encoder_input_data[i : i + 1]

predicted_sequence = decode_sequence(input_seq_onehot)

print("Input sequence (prompt):", list(input_seq_raw))
print("Predicted next sequence:", predicted_sequence)
