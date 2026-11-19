import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =====================================
# SETTINGS
# =====================================

DATASET_PATH = "dataset/faces"
MODEL_PATH = "models/deepfake_mobilenet.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# =====================================
# LOAD DATA
# =====================================

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

print("Classes:", train_dataset.class_names)

# =====================================
# PERFORMANCE
# =====================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# =====================================
# DATA AUGMENTATION
# =====================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])

# =====================================
# MOBILE NET V2
# =====================================

print("\n======================================")
print("       LOADING MOBILENETV2")
print("======================================")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# =====================================
# BUILD MODEL
# =====================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# =====================================
# COMPILE
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================
# TRAIN
# =====================================

print("\n======================================")
print("       STARTING TRANSFER LEARNING")
print("======================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# =====================================
# SAVE MODEL
# =====================================

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

# =====================================
# RESULTS
# =====================================

train_accuracy = history.history["accuracy"][-1]

validation_accuracy = history.history["val_accuracy"][-1]

print("\n======================================")
print("       TRAINING COMPLETED")
print("======================================")

print("Model saved to:")
print(os.path.abspath(MODEL_PATH))

print("\nFinal Training Accuracy:",
      round(train_accuracy * 100, 2), "%")

print("Final Validation Accuracy:",
      round(validation_accuracy * 100, 2), "%")