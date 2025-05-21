import os
import pandas as pd
from extract_features import extract_gait_features

DATASET_PATH = 'dataset'
OUTPUT_FILE = 'features.csv'

all_data = []

for person in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person)
    for file in os.listdir(person_dir):
        video_path = os.path.join(person_dir, file)
        print(f"[INFO] Processing: {video_path}")
        features = extract_gait_features(video_path)
        if features is not None:
            all_data.append([person] + list(features))

df = pd.DataFrame(all_data)
df.to_csv(OUTPUT_FILE, index=False, header=False)
print(f"[INFO] Dataset saved to {OUTPUT_FILE}")
