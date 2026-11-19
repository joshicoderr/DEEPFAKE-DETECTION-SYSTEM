import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Paths
DATASET_PATH = "dataset/faces"
MODEL_PATH = "models/deepfake_mobilenet_balanced.keras"
RESULTS_PATH = "results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs(RESULTS_PATH, exist_ok=True)

print("Loading images...")

# Get image paths
real_path = os.path.join(DATASET_PATH, "real")
fake_path = os.path.join(DATASET_PATH, "fake")

real_images = [
    os.path.join(real_path, f)
    for f in os.listdir(real_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

fake_images = [
    os.path.join(fake_path, f)
    for f in os.listdir(fake_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Labels: REAL = 0, FAKE = 1
image_paths = real_images + fake_images
labels = [0] * len(real_images) + [1] * len(fake_images)

print("REAL images:", len(real_images))
print("FAKE images:", len(fake_images))
print("Total images:", len(image_paths))

# Create balanced/random validation split
X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("Evaluation images:", len(X_test))

# Load MobileNetV2 model
print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

# Load images
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
)

dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Predictions
print("Making predictions...")

predictions = model.predict(dataset)

predicted_labels = (predictions.flatten() >= 0.5).astype(int)

# Classification report
print("\n======================================")
print("       CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        predicted_labels,
        target_names=["REAL", "FAKE"]
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, predicted_labels)

print("\n======================================")
print("       CONFUSION MATRIX")
print("======================================")

print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Deepfake Detection Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["REAL", "FAKE"])
plt.yticks([0, 1], ["REAL", "FAKE"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

output_file = os.path.join(
    RESULTS_PATH,
    "confusion_matrix_mobilenet.png"
)

plt.savefig(output_file)

print("\nConfusion matrix saved to:")
print(output_file)