import cv2
import torch
import joblib
import numpy as np
import mediapipe as mp
from pose_extractor.extract_features import calculate_angle
from train.model import GaitRecognitionModel

def recognize_and_save(video_path, output_path='outputs/output_video.mp4', threshold=0.8):
    # Load scaler and encoder
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/encoder.pkl')

    # Load model
    input_dim = scaler.mean_.shape[0]
    model = GaitRecognitionModel(input_dim=input_dim, num_classes=len(encoder.classes_))
    model.load_state_dict(torch.load('models/gait_model.pth'))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    frame_buffer = []
    frame_count = 0
    sample_rate = 5

    last_label = "Detecting..."
    last_prob = 0.0
    last_bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        display_frame = frame.copy()

        frame_count += 1
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            xs = [landmark.x for landmark in results.pose_landmarks.landmark]
            ys = [landmark.y for landmark in results.pose_landmarks.landmark]
            min_x = int(min(xs) * width)
            max_x = int(max(xs) * width)
            min_y = int(min(ys) * height)
            max_y = int(max(ys) * height)
            current_bbox = (min_x, min_y, max_x, max_y)

            if frame_count % sample_rate == 0:
                # Every few frames - try to predict
                keypoints = results.pose_landmarks.landmark
                frame_feats = []
                important_landmarks = [11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
                for idx in important_landmarks:
                    frame_feats.append(keypoints[idx].x)
                    frame_feats.append(keypoints[idx].y)
                frame_buffer.append(frame_feats)


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
                        last_label = f"{best_label} ({best_prob*100:.1f}%)"
                        last_color = (0, 255, 0)  # Green
                    else:
                        last_label = f"Unknown ({best_prob*100:.1f}%)"
                        last_color = (0, 0, 255)  # Red

                    last_bbox = current_bbox
                    frame_buffer = []

        # Always draw last prediction
        if last_bbox:
            min_x, min_y, max_x, max_y = last_bbox
            cv2.rectangle(display_frame, (min_x-20, min_y-20), (max_x+20, max_y+20), last_color, 3)
            cv2.putText(display_frame, last_label, (min_x, min_y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)

        out.write(display_frame)
        cv2.imshow('Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"[INFO] Output saved successfully at {output_path}")

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

if __name__ == "__main__":
    recognize_and_save('dataset/person1/walk1.mp4')
