import json
import os

DATASET_PATH = "dataset/train_sample_videos"
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.json")

if not os.path.exists(METADATA_PATH):
    print("ERROR: metadata.json was not found!")
    print("Expected location:")
    print(os.path.abspath(METADATA_PATH))
    exit()

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

real_count = 0
fake_count = 0

print("======================================")
print("       DEEPFAKE DATASET CHECK")
print("======================================")

for filename, information in metadata.items():
    label = information.get("label")

    if label == "REAL":
        real_count += 1
    elif label == "FAKE":
        fake_count += 1

print("Total videos :", len(metadata))
print("REAL videos  :", real_count)
print("FAKE videos  :", fake_count)

print("\nFirst 10 videos:")

count = 0

for filename, information in metadata.items():
    print(filename, "→", information.get("label"))
    count += 1

    if count == 10:
        break

print("\n======================================")
print("Dataset check completed!")
print("======================================")