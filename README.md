# Speech-Emotion-Recognition
# Speech Emotion Recognition using 1D-CNN

A deep learning project for classifying human speech into eight emotional categories using **Mel-Frequency Cepstral Coefficients (MFCCs)** and a **1D Convolutional Neural Network (1D-CNN)**.

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset and includes audio preprocessing, MFCC feature extraction, model training, emotion prediction, evaluation, and CNN activation visualization.

**Project Status:** Completed / Portfolio Project
**Language:** Python
**Framework:** TensorFlow / Keras

---

## Overview

Speech contains emotional information beyond the words being spoken. Changes in pitch, intensity, and spectral characteristics can provide useful signals for identifying emotional states.

This project explores whether a lightweight 1D-CNN can learn meaningful patterns from speech features and classify audio into eight emotion categories.

The system processes audio using **40 MFCC features**, trains a two-layer 1D-CNN, and provides tools for prediction, evaluation, and visualization of intermediate CNN activations.

---

## Features

* 🎙️ Speech emotion classification
* 🎵 40 MFCC feature extraction
* 🧠 1D Convolutional Neural Network
* 📊 Accuracy and classification-report evaluation
* 🔲 Confusion matrix visualization
* 🔍 CNN activation / feature visualization
* 💾 Saved trained Keras model
* 🧪 Individual audio-file prediction

---

## Emotions

The model classifies speech into eight categories:

| Label | Emotion   |
| ----- | --------- |
| 01    | Neutral   |
| 02    | Calm      |
| 03    | Happy     |
| 04    | Sad       |
| 05    | Angry     |
| 06    | Fearful   |
| 07    | Disgust   |
| 08    | Surprised |

These labels follow the RAVDESS filename convention used by the preprocessing pipeline.

---

## Dataset

This project uses the **RAVDESS dataset**:

> Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

RAVDESS contains emotional speech recordings performed by professional actors and provides eight emotion categories used in this project.

### Dataset Download

The dataset is **not included in this repository** because of its size and dataset distribution considerations.

Download RAVDESS from its official source:

https://zenodo.org/records/1188976

After downloading the audio speech files, place the dataset inside the project directory using the following structure:

```text
Speech-Emotion-Recognition/
│
├── data/
│   ├── Actor_01/
│   ├── Actor_02/
│   ├── Actor_03/
│   └── ...
│
├── 1_process_data.py
├── 2_train_model.py
├── 3_predict.py
├── 4_evaluate_accuracy.py
├── 5_visualize_layers.py
├── README.md
├── LICENSE
└── .gitignore
```

The preprocessing script expects the dataset under `./data/`.

---

## Methodology

The project follows this pipeline:

```text
RAVDESS Audio
      ↓
Audio Preprocessing
      ↓
MFCC Feature Extraction
      ↓
40 MFCC Features
      ↓
1D-CNN
      ↓
Emotion Classification
      ↓
Evaluation & Visualization
```

### 1. Audio preprocessing

Audio files are loaded using `librosa` at a sampling rate of **22,050 Hz**. The current implementation processes up to three seconds of audio with a 0.5-second offset.

### 2. MFCC extraction

The system extracts **40 Mel-Frequency Cepstral Coefficients (MFCCs)** from each audio file.

The MFCC representation is then averaged across the time dimension to produce a 40-feature representation for each audio sample.

### 3. 1D-CNN architecture

The model consists of two convolutional layers:

```text
Input: 40 MFCC features

Conv1D
64 filters
Kernel size = 5
ReLU
MaxPooling
Dropout = 0.2

        ↓

Conv1D
128 filters
Kernel size = 5
ReLU
MaxPooling
Dropout = 0.2

        ↓

Flatten

        ↓

Dense
64 units
ReLU

        ↓

Dense
8 units
Softmax
```

The implementation uses TensorFlow/Keras and trains with the Adam optimizer and categorical cross-entropy loss.

---

## Training

The model is trained for **50 epochs** with a batch size of **32**.

The current implementation uses an 80/20 train-test split with `random_state=42`.
After training, the model is saved as:

```text
emotion_model.keras
```

---

## Results

The project documentation reports a test accuracy of:

### **93.40%**

The original project also compares the fine-tuned model with an earlier baseline using 13 MFCC features and a shallower CNN architecture.

| Model    | MFCC Features | Architecture     | Reported Accuracy |
| -------- | ------------: | ---------------- | ----------------: |
| Baseline |            13 | Shallow 1D-CNN   |            85.40% |
| Proposed |            40 | Two-layer 1D-CNN |            93.40% |

The reported baseline and proposed results are documented in the accompanying technical report.

> **Note:** The reported accuracy should be interpreted in the context of the current random train-test evaluation procedure. Further speaker-independent evaluation would make the experimental results more robust.

---

## Evaluation

The project includes a dedicated evaluation script that calculates:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

The evaluation script uses `classification_report` from scikit-learn and generates a confusion matrix showing where the model succeeds and where different emotions are confused.

### Example evaluation output

```text
Final Model Accuracy: 93.40%

Detailed Classification Report:
Precision
Recall
F1-score
Support
```

The exact class-level precision, recall, and F1-score values depend on the evaluation run.

---

## Emotion Prediction

The prediction script can load the trained model and classify an individual RAVDESS audio file.

The process is:

```text
Audio File
    ↓
22,050 Hz preprocessing
    ↓
40 MFCCs
    ↓
Feature averaging
    ↓
1D-CNN
    ↓
Predicted Emotion
```

## The prediction script also compares the predicted emotion with the emotion encoded in the RAVDESS filename.

## Explainable AI / CNN Visualization

To investigate what the CNN learns from the MFCC representation, the project includes an activation visualization script.

The script creates an intermediate Keras model that outputs the activation of the first convolutional layer and visualizes the resulting activation map as a heatmap.
This provides a visual representation of the patterns detected by the convolutional filters.

> This visualization should be considered a feature-activation analysis rather than a complete explanation of the model's final decision.

---

## Project Structure

```text
Speech-Emotion-Recognition/
│
├── 1_process_data.py
│   └── Loads RAVDESS audio and extracts 40 MFCC features
│
├── 2_train_model.py
│   └── Builds and trains the 1D-CNN
│
├── 3_predict.py
│   └── Predicts the emotion of an audio file
│
├── 4_evaluate_accuracy.py
│   └── Calculates accuracy, classification metrics,
│       and generates a confusion matrix
│
├── 5_visualize_layers.py
│   └── Visualizes intermediate CNN activations
│
├── X_data.npy
│   └── Processed MFCC features
│
├── y_data.npy
│   └── Emotion labels
│
├── classes.npy
│   └── Encoded emotion classes
│
├── emotion_model.keras
│   └── Trained neural network
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Rayan-Shahid/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

Install the required Python packages:

```bash
pip install numpy pandas librosa tensorflow scikit-learn matplotlib seaborn
```

Python 3.9+ is recommended.

---

## Usage

### Step 1 — Download the dataset

Download the RAVDESS dataset and place the audio files inside:

```text
./data/
```

### Step 2 — Process the audio

Run:

```bash
python 1_process_data.py
```

This extracts the MFCC features and generates:

```text
X_data.npy
y_data.npy
```

### Step 3 — Train the model

Run:

```bash
python 2_train_model.py
```

The trained model will be saved as:

```text
emotion_model.keras
```

The class labels are also saved as:

```text
classes.npy
```

### Step 4 — Make a prediction

Run:

```bash
python 3_predict.py
```

The script selects an audio file from the dataset and displays the actual and predicted emotion.

### Step 5 — Evaluate the model

Run:

```bash
python 4_evaluate_accuracy.py
```

This produces the model accuracy, classification report, and confusion matrix.

### Step 6 — Visualize CNN activations

Run:

```bash
python 5_visualize_layers.py
```

This generates a visualization of the activations produced by the first convolutional layer.

---

## Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Librosa**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**

---

## Limitations

This project is primarily an experimental and educational implementation of Speech Emotion Recognition.

Current limitations include:

* Evaluation currently uses a random 80/20 train-test split.
* Speaker-independent evaluation has not yet been established.
* The model is evaluated on RAVDESS rather than diverse real-world speech.
* The current feature representation averages MFCCs over time, which removes some temporal information.
* Real-world background noise and spontaneous speech may significantly affect performance.
* The activation heatmap provides intermediate feature visualization rather than a complete explanation of the final prediction.

---

## Future Work

Potential improvements include:

* Speaker-independent train/test evaluation
* Cross-validation
* Data augmentation
* Testing on noisy and real-world speech
* Comparing 1D-CNN with LSTM, CNN-LSTM, and transformer-based approaches
* Preserving more temporal information from MFCC sequences
* Hyperparameter optimization
* Additional explainability techniques
* Real-time speech emotion recognition
* Deployment as a web or desktop application

---

## References

The research background, methodology, and related work are documented in the project's accompanying technical report.

Key references include work on:

* Speech Emotion Recognition
* RAVDESS
* MFCC feature extraction
* 1D convolutional neural networks
* Explainable AI

---

## Author

**Rayan Shahid**

Computer Science

GitHub:
https://github.com/Rayan-Shahid

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Disclaimer

This project is intended for educational and research purposes. Emotion classification from speech should not be treated as a definitive assessment of a person's psychological or mental state.

