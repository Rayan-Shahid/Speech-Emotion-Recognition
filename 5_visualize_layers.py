import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import os
import random

# 1. SETUP
# Load the model
model = tf.keras.models.load_model('emotion_model.keras')

# --- THE FIX FOR KERAS 3 ERROR ---
# We force the model to "build" by passing one fake piece of data through it.
# This initializes the input tensor so we can access it.
dummy_data = np.zeros((1, 40, 1))
_ = model.predict(dummy_data, verbose=0)
# ---------------------------------

# 2. CREATE A "FEATURE EXTRACTOR" MODEL
# We use 'model.inputs' (plural) which is safer for newer Keras versions
layer_name = model.layers[0].name
print(f"Visualizing layer: {layer_name}")

intermediate_layer_model = tf.keras.models.Model(
    inputs=model.inputs,     # Changed from model.input to model.inputs
    outputs=model.layers[0].output
)

# 3. GET A RANDOM SAMPLE
data_folder = "./data/"
all_files = []
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith(".wav"):
            all_files.append(os.path.join(root, file))

if not all_files:
    print("Error: No .wav files found in ./data/")
    exit()

random_file = random.choice(all_files)
print(f"Analyzing file: {os.path.basename(random_file)}")

# Process the audio
y, sr = librosa.load(random_file, sr=22050, duration=3, offset=0.5)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc_processed = np.mean(mfcc.T, axis=0)
features = np.expand_dims(mfcc_processed, axis=0)
features = np.expand_dims(features, axis=2) # Shape: (1, 40, 1)

# 4. GET ACTIVATIONS
activations = intermediate_layer_model.predict(features)
activation_map = np.squeeze(activations) 

# 5. PLOT THE HEATMAP
plt.figure(figsize=(12, 6))
plt.imshow(activation_map.T, aspect='auto', cmap='viridis', origin='lower')
plt.title(f"What the CNN 'Sees' in: {os.path.basename(random_file)}")
plt.xlabel('Time / Feature Steps')
plt.ylabel('Filter Index (0-63)')
plt.colorbar(label='Activation Strength')
plt.show()