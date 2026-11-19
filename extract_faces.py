import cv2
import json
import os

# Dataset location
DATASET_PATH = "dataset/train_sample_videos"
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.json")

# Where extracted faces will be saved
OUTPUT_PATH = "dataset/faces"

# Number of frames to take from each video
FRAMES_PER_VIDEO = 5

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create output folders
os.makedirs(os.path.join(OUTPUT_PATH, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "fake"), exist_ok=True)

# Load metadata
with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

print("======================================")
print("       FACE EXTRACTION STARTED")
print("======================================")

video_count = 0
face_count = 0

for filename, information in metadata.items():

    label = information.get("label")

    if label not in ["REAL", "FAKE"]:
        continue

    video_path = os.path.join(DATASET_PATH, filename)

    if not os.path.exists(video_path):
        print("Video not found:", filename)
        continue

    video = cv2.VideoCapture(video_path)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        video.release()
        continue

    # Select frames evenly throughout the video
    frame_numbers = [
        int(total_frames * i / FRAMES_PER_VIDEO)
        for i in range(FRAMES_PER_VIDEO)
    ]

    saved_from_video = 0

    for frame_number in frame_numbers:

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = video.read()

        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            continue

        # Use the largest detected face
        face = max(faces, key=lambda x: x[2] * x[3])

        x, y, w, h = face

        face_image = frame[y:y+h, x:x+w]

        # Resize face
        face_image = cv2.resize(face_image, (224, 224))

        label_folder = "real" if label == "REAL" else "fake"

        output_name = (
            os.path.splitext(filename)[0]
            + "_"
            + str(frame_number)
            + ".jpg"
        )

        output_file = os.path.join(
            OUTPUT_PATH,
            label_folder,
            output_name
        )

        cv2.imwrite(output_file, face_image)

        face_count += 1
        saved_from_video += 1

    video.release()

    video_count += 1

    print(
        f"Processed {video_count}/{len(metadata)}: "
        f"{filename} → {label} → {saved_from_video} faces"
    )

print("\n======================================")
print("       FACE EXTRACTION COMPLETED")
print("======================================")

print("Videos processed :", video_count)
print("Faces extracted  :", face_count)

print("\nSaved in:")
print("REAL :", os.path.abspath(os.path.join(OUTPUT_PATH, "real")))
print("FAKE :", os.path.abspath(os.path.join(OUTPUT_PATH, "fake")))