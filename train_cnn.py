import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset/faces"
MODEL_PATH = "models/deepfake_cnn.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# ==============================
# LOAD DATASET
# ==============================

print("======================================")
print("       LOADING FACE DATASET")
print("======================================")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["real", "fake"]
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["real", "fake"]
)

print("\nClass names:", train_dataset.class_names)

# Improve loading speed
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])

# ==============================
# CNN MODEL
# ==============================

print("\n======================================")
print("          BUILDING CNN MODEL")
print("======================================")

model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    # Data augmentation
    data_augmentation,

    # Normalize pixel values
    layers.Rescaling(1.0 / 255),

    # CNN Layer 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # CNN Layer 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # CNN Layer 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # CNN Layer 4
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convert feature maps to vector
    layers.Flatten(),

    # Fully connected layers
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Output
    layers.Dense(1, activation="sigmoid")
])

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# TRAIN MODEL
# ==============================

print("\n======================================")
print("          STARTING TRAINING")
print("======================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# ==============================
# SAVE MODEL
# ==============================

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

print("\n======================================")
print("          TRAINING COMPLETED")
print("======================================")

print("Model saved to:")
print(os.path.abspath(MODEL_PATH))

# ==============================
# FINAL ACCURACY
# ==============================

train_accuracy = history.history["accuracy"][-1]
validation_accuracy = history.history["val_accuracy"][-1]

print("\nFinal Training Accuracy:",
      round(train_accuracy * 100, 2), "%")

print("Final Validation Accuracy:",
      round(validation_accuracy * 100, 2), "%")