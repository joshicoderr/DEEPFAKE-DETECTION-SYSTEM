import cv2
import os
import numpy as np
import tensorflow as tf

# ==========================================
# PATHS
# ==========================================

VIDEO_FOLDER = "dataset/train_sample_videos"
MODEL_PATH = "models/deepfake_mobilenet_balanced.keras"

IMG_SIZE = (224, 224)
FRAME_INTERVAL = 10

# ==========================================
# FIND VIDEOS
# ==========================================

videos = [
    f for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith(".mp4")
]

if len(videos) == 0:
    print("No videos found in:")
    print(os.path.abspath(VIDEO_FOLDER))
    exit()

print("\n======================================")
print("       DEEPFAKE VIDEO DETECTOR")
print("======================================")

print("\nAvailable videos:")

for i, video in enumerate(videos, start=1):
    print(f"{i}. {video}")

# ==========================================
# SELECT VIDEO
# ==========================================

while True:

    try:
        choice = int(input("\nEnter video number: "))

        if 1 <= choice <= len(videos):
            break

        print("Please enter a valid number.")

    except ValueError:
        print("Please enter a number.")

selected_video = videos[choice - 1]

VIDEO_PATH = os.path.join(
    VIDEO_FOLDER,
    selected_video
)

print("\nSelected video:")
print(selected_video)

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading DeepFake detection model...")

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# OPEN VIDEO
# ==========================================

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():

    print("\nERROR: Could not open video.")
    print("Video path:")
    print(os.path.abspath(VIDEO_PATH))

    exit()

total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = video.get(cv2.CAP_PROP_FPS)

print("\nTotal frames:", total_frames)
print("FPS:", fps)

# ==========================================
# PROCESS VIDEO
# ==========================================

predictions = []
frame_number = 0

print("\nProcessing video...")

while True:

    success, frame = video.read()

    if not success:
        break

    # Process every 10th frame
    if frame_number % FRAME_INTERVAL == 0:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) > 0:

            # Select largest face
            face = max(
                faces,
                key=lambda x: x[2] * x[3]
            )

            x, y, w, h = face

            face_image = frame[
                y:y+h,
                x:x+w
            ]

            face_image = cv2.resize(
                face_image,
                IMG_SIZE
            )

            face_image = cv2.cvtColor(
                face_image,
                cv2.COLOR_BGR2RGB
            )

            face_image = np.array(
                face_image,
                dtype=np.float32
            )

            # MobileNetV2 preprocessing
            face_image = tf.keras.applications.mobilenet_v2.preprocess_input(
                face_image
            )

            face_image = np.expand_dims(
                face_image,
                axis=0
            )

            # Prediction
            prediction = model.predict(
                face_image,
                verbose=0
            )[0][0]

            predictions.append(
                float(prediction)
            )

    frame_number += 1

video.release()

# ==========================================
# FINAL RESULT
# ==========================================

print("\n======================================")
print("       DEEPFAKE DETECTION RESULT")
print("======================================")

if len(predictions) == 0:

    print("No faces were detected.")
    print("Try another video.")

else:

    average_probability = np.mean(
        predictions
    )

    fake_percentage = (
        average_probability * 100
    )

    real_percentage = (
        (1 - average_probability) * 100
    )

    print(
        "Frames with detected faces:",
        len(predictions)
    )

    print(
        "REAL probability:",
        round(float(real_percentage), 2),
        "%"
    )

    print(
        "FAKE probability:",
        round(float(fake_percentage), 2),
        "%"
    )

    if average_probability >= 0.5:

        print("\nFINAL RESULT: FAKE / DEEPFAKE")

    else:

        print("\nFINAL RESULT: REAL")

print("\n======================================")