import sys
import os
import cv2
import torch
import joblib
import numpy as np
import mediapipe as mp

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose_extractor.extract_features import calculate_angle
from train.model import GaitRecognitionModel


def run_realtime_gait_recognition(threshold=0.8):
    # Load model and preprocessors
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')
    input_dim = scaler.mean_.shape[0]

    model = GaitRecognitionModel(input_dim=input_dim, num_classes=len(encoder.classes_))
    model.load_state_dict(torch.load('models/gait_model.pth', map_location=torch.device('cpu')))
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible.")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    frame_buffer = []
    frame_count = 0
    sample_rate = 5
    last_label = "Detecting..."
    last_color = (255, 255, 255)
    last_bbox = None

    # Fullscreen window
    cv2.namedWindow('Real-Time Gait Recognition', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Real-Time Gait Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip for natural view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        display_frame = frame.copy()
        frame_count += 1

        if results.pose_landmarks:
            # Draw pose
            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            keypoints = results.pose_landmarks.landmark
            xs = [kp.x for kp in keypoints]
            ys = [kp.y for kp in keypoints]
            min_x = int(min(xs) * frame.shape[1])
            max_x = int(max(xs) * frame.shape[1])
            min_y = int(min(ys) * frame.shape[0])
            max_y = int(max(ys) * frame.shape[0])
            current_bbox = (min_x, min_y, max_x, max_y)

            if frame_count % sample_rate == 0:
                frame_feats = []
                for kp in keypoints:
                    frame_feats.append(kp.x)
                    frame_feats.append(kp.y)

                frame_buffer.append(frame_feats)

                if len(frame_buffer) >= 5:
                    feature_sequence = np.array(frame_buffer)
                    mean_features = np.mean(feature_sequence, axis=0)
                    features = scaler.transform([mean_features])
                    features = torch.tensor(features, dtype=torch.float32)

                    output = model(features)
                    probs = torch.softmax(output, dim=1).detach().numpy()[0]
                    best_idx = np.argmax(probs)
                    best_label = encoder.inverse_transform([best_idx])[0]
                    best_prob = probs[best_idx]

                    if best_prob >= threshold:
                        last_label = f"{best_label} ({best_prob * 100:.1f}%)"
                        last_color = (0, 255, 0)
                    else:
                        last_label = f"Unknown ({best_prob * 100:.1f}%)"
                        last_color = (0, 0, 255)

                    last_bbox = current_bbox
                    frame_buffer = []

        if last_bbox:
            min_x, min_y, max_x, max_y = last_bbox
            cv2.rectangle(display_frame, (min_x - 20, min_y - 20), (max_x + 20, max_y + 20), last_color, 3)
            cv2.putText(display_frame, last_label, (min_x, min_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)

        cv2.imshow('Real-Time Gait Recognition', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:  # Quit on 'q' or 'Esc'
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    run_realtime_gait_recognition()
