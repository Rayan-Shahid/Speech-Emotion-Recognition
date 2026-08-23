import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 1. LOAD DATA
print("Loading data...")
X = np.load('X_data.npy')
y = np.load('y_data.npy')

# 2. PREPARE LABELS
# Convert words ('angry', 'happy') to numbers (0, 1, 2...)
lb = LabelEncoder()
y_cat = to_categorical(lb.fit_transform(y))
np.save('classes.npy', lb.classes_) # Save label names for later

# 3. RESHAPE FOR CNN
# CNN expects 3D input: (Batch, Features, TimeSteps/Channels)
# Our features are 40, so we reshape to (Rows, 40, 1)
X = np.expand_dims(X, axis=2)

# 4. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 5. BUILD MODEL (1D CNN)
model = Sequential()

# Layer 1: Convolution
model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(40, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Layer 2: Convolution
model.add(Conv1D(128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Flatten and Output
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='softmax')) # 8 Emotions

# 6. TRAIN
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 7. SAVE MODEL
model.save('emotion_model.keras')
print("Model saved as 'emotion_model.keras'")





