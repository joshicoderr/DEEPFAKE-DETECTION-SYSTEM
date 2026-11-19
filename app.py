import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/deepfake_mobilenet_balanced.keras"

IMG_SIZE = (224, 224)
FRAME_INTERVAL = 10

# ==========================================
# PAGE
# ==========================================

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 DeepFake Detection System")
st.write("Upload a video to check whether it is REAL or FAKE.")

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )

model = load_model()

# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# UPLOAD VIDEO
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

# ==========================================
# PROCESS VIDEO
# ==========================================

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("🔍 Detect DeepFake"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            video_path = temp_file.name

        st.info("Processing video... Please wait.")

        video = cv2.VideoCapture(video_path)

        predictions = []
        frame_number = 0

        progress = st.progress(0)

        total_frames = int(
            video.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        while True:

            success, frame = video.read()

            if not success:
                break

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

                    face_image = tf.keras.applications.mobilenet_v2.preprocess_input(
                        face_image
                    )

                    face_image = np.expand_dims(
                        face_image,
                        axis=0
                    )

                    prediction = model.predict(
                        face_image,
                        verbose=0
                    )[0][0]

                    predictions.append(
                        float(prediction)
                    )

            frame_number += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_number / total_frames,
                        1.0
                    )
                )

        video.release()

        os.remove(video_path)

        # ==========================================
        # RESULT
        # ==========================================

        if len(predictions) == 0:

            st.error(
                "No faces were detected in the video."
            )

        else:

            average_probability = np.mean(
                predictions
            )

            fake_probability = (
                average_probability * 100
            )

            real_probability = (
                (1 - average_probability) * 100
            )

            st.success(
                f"Processed {len(predictions)} face frames."
            )

            st.subheader("Prediction")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "REAL Probability",
                    f"{real_probability:.2f}%"
                )

            with col2:
                st.metric(
                    "FAKE Probability",
                    f"{fake_probability:.2f}%"
                )

            if average_probability >= 0.5:

                st.error(
                    "🚨 FINAL RESULT: FAKE / DEEPFAKE"
                )

            else:

                st.success(
                    "✅ FINAL RESULT: REAL"
                )