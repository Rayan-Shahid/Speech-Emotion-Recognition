import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOAD DATA & MODEL
print("Loading data and model...")
X = np.load('X_data.npy')
y = np.load('y_data.npy') # These are the text labels like 'happy', 'sad'
model = tf.keras.models.load_model('emotion_model.keras')

# 2. RE-CREATE THE DATA SPLIT
# We must encode labels to numbers first (same as before)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

lb = LabelEncoder()
y_cat = to_categorical(lb.fit_transform(y)) # Convert to [0, 0, 1...] format

# RESHAPE X (Same as training)
X = np.expand_dims(X, axis=2)

# SPLIT (Must use random_state=42 to match training split!)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 3. CALCULATE OVERALL ACCURACY
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\n--------------------------------------------------")
print(f"🏆 FINAL MODEL ACCURACY: {accuracy * 100:.2f}%")
print("--------------------------------------------------\n")

# 4. GENERATE CONFUSION MATRIX
# Get predictions for all test files
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Get the label names (neutral, happy, etc.)
label_names = lb.classes_

# Create the Matrix
cm = confusion_matrix(true_classes, predicted_classes)

# 5. PLOT IT
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Emotion')
plt.ylabel('Actual Emotion')
plt.title('Confusion Matrix: Where does the model get confused?')
plt.show()

# 6. DETAILED REPORT (Precision/Recall)
print("\nDetailed Classification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=label_names))