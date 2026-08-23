import librosa
import numpy as np
import os
import pandas as pd

# 1. DEFINE PATHS
DATA_PATH = "./data/"  # Folder containing Actor_01, Actor_02...

# 2. DEFINE EMOTIONS (Based on RAVDESS filename documentation)
# Filename example: 03-01-06-01-02-01-12.wav
# The 3rd part (06) is the emotion.
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    # Load audio (downsample to 22050 Hz for speed)
    y, sr = librosa.load(file_path, sr=22050, duration=3, offset=0.5)
    
    # Extract MFCCs (40 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Average over time (result shape: 40,)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    return mfcc_processed

# 3. MAIN LOOP
data = []
labels = []

print("Starting data processing... this might take 2-3 minutes.")

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            # Get the label from the filename (3rd part)
            part = file.split('-')[2]
            label = emotion_map.get(part)
            
            if label:
                file_path = os.path.join(root, file)
                
                # Extract features
                features = extract_features(file_path)
                
                data.append(features)
                labels.append(label)

# 4. SAVE TO FILE (So we don't have to wait next time)
X = np.array(data)
y = np.array(labels)

np.save('X_data.npy', X)
np.save('y_data.npy', y)

print(f"Success! Processed {len(X)} files.")
print("Data saved to X_data.npy and y_data.npy")