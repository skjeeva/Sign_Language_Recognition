import cv2
import numpy as np
from keras.models import load_model
import json
import os

# Load the trained model
model = load_model("asl_model.h5")

# Load class labels from saved class_indices.json
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Convert dictionary {class: index} -> [class0, class1, ...]
labels = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

print("Camera initialized. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Draw rectangle for ROI
    x1, y1, x2, y2 = 100, 100, 324, 324
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # ✅ Resize to 64x64 and keep RGB (3 channels)
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float32") / 255.0       # Normalize to [0, 1]
    roi = np.expand_dims(roi, axis=0)         # shape: (1, 64, 64, 3)

    # Predict
    predictions = model.predict(roi)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index] * 100
    class_label = labels[class_index]

    # Show prediction only if confidence is above threshold
    if confidence > 70:
        text = f"{class_label} ({confidence:.1f}%)"
    else:
        text = "..."

    # Display result
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Sign Language Detection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
