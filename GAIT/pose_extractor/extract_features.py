import cv2
import mediapipe as mp
import numpy as np

def extract_gait_features(video_path, sample_rate=5):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return None

    features = []
    frame_count = 0

    important_landmarks = [
        11, 12, 23, 24, 25, 26, 27, 28, 31, 32  # shoulders, hips, knees, ankles, heels
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sample_rate != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            keypoints = results.pose_landmarks.landmark
            frame_feats = []
            for idx in important_landmarks:
                frame_feats.append(keypoints[idx].x)
                frame_feats.append(keypoints[idx].y)
            features.append(frame_feats)

    cap.release()
    pose.close()

    if not features:
        return None

    features = np.array(features)
    mean_features = np.mean(features, axis=0)
    return mean_features

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)
