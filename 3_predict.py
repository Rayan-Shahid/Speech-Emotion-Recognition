import librosa
import numpy as np
import tensorflow as tf
import os
import random
from sklearn.preprocessing import LabelEncoder

# 1. SETUP
model = tf.keras.models.load_model('emotion_model.keras')
classes = np.load('classes.npy')

# Map filename codes to emotions (RAVDESS standard)
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def predict_emotion(audio_path):
    # Load and process audio
    y, sr = librosa.load(audio_path, sr=22050, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    
    # Reshape for CNN
    features = np.expand_dims(mfcc_processed, axis=0)
    features = np.expand_dims(features, axis=2)
    
    # Predict
    prediction = model.predict(features, verbose=0)
    predicted_index = np.argmax(prediction)
    return classes[predicted_index]

# 2. FIND A RANDOM FILE TO TEST
data_folder = "./data/"
all_files = []

# Walk through folders to find all .wav files
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith(".wav"):
            all_files.append(os.path.join(root, file))

if len(all_files) == 0:
    print("Error: No files found in ./data/ folder.")
else:
    # Pick a random file
    random_file = random.choice(all_files)
    
    # 3. GET ACTUAL TRUTH (FROM FILENAME)
    filename = os.path.basename(random_file)
    actual_code = filename.split('-')[2] # The 3rd part is the emotion
    actual_emotion = emotion_map.get(actual_code)

    print(f"Testing File: {filename}")
    print("--------------------------------------------------")
    
    # 4. GET PREDICTION
    predicted_emotion = predict_emotion(random_file)

    # 5. COMPARE
    print(f"✅ ACTUAL Emotion:    {actual_emotion.upper()}")
    print(f"🤖 MODEL Prediction:  {predicted_emotion.upper()}")
    
    if actual_emotion == predicted_emotion:
        print("\n🎉 SUCCESS! The model got it right!")
    else:
        print("\n❌ WRONG! The model made a mistake.")