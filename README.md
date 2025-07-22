### 📄 `README.md`

```markdown
# 🖐️ Sign Language Recognition System (ASL)

This project recognizes American Sign Language (ASL) gestures in real-time using hand tracking (MediaPipe) and a trained deep learning model (Keras + TensorFlow). It detects hand landmarks and classifies gestures from webcam input.

---

## 🚀 Features

- Real-time hand gesture tracking using MediaPipe
- Custom CNN trained on ASL alphabet dataset
- Live camera prediction using saved `.h5` model
- Easy training and validation from image dataset folders

---

## 📁 Project Structure

```

.
├── training.py               # Train ASL CNN model
├── sign\_language.py          # Real-time prediction using webcam
├── hand\_gesture\_detection.py # Landmark tracking only
├── class\_indices.json        # Mapping of label index to character
├── asl\_model.h5              # Trained model
├── requirements.txt          # Required packages
├── README.md                 # This file
└── asl\_alphabet\_train/       # Training dataset (A-Z folders)

```

---

## 🧠 Model Training

Make sure you have your dataset structured like this:

```

asl\_alphabet\_train/
├── A/
├── B/
├── ...
└── Z/

````

Then run:

```bash
python training.py
````

This will:

* Train a CNN using data augmentation
* Save the model as `asl_model.h5`
* Generate `class_indices.json` for label mapping

---

## 🤖 Real-time Gesture Recognition

Once trained:

```bash
python sign_language.py
```

* Opens your webcam
* Detects hand gestures
* Predicts the class label using the trained model

To exit, press **`q`**.

---

## 🖐️ View Only Hand Landmarks

If you just want to test hand detection:

```bash
python hand_gesture_detection.py
```

---

## 📦 Installation

Create a virtual environment and install all dependencies:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔧 Requirements

* Python 3.8+
* TensorFlow 2.x
* OpenCV
* MediaPipe
* Keras

---

## 🙌 Credits

* [MediaPipe](https://github.com/google/mediapipe) for hand detection
* TensorFlow/Keras for model training
* ASL Alphabet Dataset for training

---

## 🧠 To-Do

* [ ] Add GUI interface
* [ ] Support Indian Sign Language (ISL)
* [ ] Improve model accuracy using transfer learning

## 👨‍💻 Developed by Jeeva