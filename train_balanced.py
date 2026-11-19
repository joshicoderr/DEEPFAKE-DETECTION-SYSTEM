import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset/faces"
MODEL_PATH = "models/deepfake_mobilenet_balanced.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

random.seed(42)

# ==============================
# GET IMAGE FILES
# ==============================

real_dir = os.path.join(DATASET_PATH, "real")
fake_dir = os.path.join(DATASET_PATH, "fake")

real_images = [
    os.path.join(real_dir, f)
    for f in os.listdir(real_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

fake_images = [
    os.path.join(fake_dir, f)
    for f in os.listdir(fake_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Original REAL images:", len(real_images))
print("Original FAKE images:", len(fake_images))

# ==============================
# BALANCE DATASET
# ==============================

# Use the same number from both classes
number = min(len(real_images), len(fake_images))

random.shuffle(real_images)
random.shuffle(fake_images)

real_images = real_images[:number]
fake_images = fake_images[:number]

print("Balanced REAL images:", len(real_images))
print("Balanced FAKE images:", len(fake_images))

# ==============================
# COMBINE DATA
# ==============================

image_paths = real_images + fake_images
labels = [0] * len(real_images) + [1] * len(fake_images)

# Split while keeping classes balanced
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))

# ==============================
# CREATE DATASET
# ==============================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)

    return image, label


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(1000)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

# ==============================
# MOBILE NET V2
# ==============================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ==============================
# MODEL
# ==============================

model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# TRAIN
# ==============================

print("\n======================================")
print("   BALANCED MOBILE NET TRAINING")
print("======================================")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# ==============================
# SAVE
# ==============================

model.save(MODEL_PATH)

print("\n======================================")
print("       TRAINING COMPLETED")
print("======================================")

print("Model saved to:")
print(MODEL_PATH)