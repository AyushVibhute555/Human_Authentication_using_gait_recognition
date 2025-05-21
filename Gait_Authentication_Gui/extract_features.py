import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def angle_between(v1, v2):
    """Returns the angle in radians between vectors v1 and v2"""
    return np.arctan2(np.linalg.det([v1,v2]), np.dot(v1,v2))

def calculate_advanced_joint_angles(landmarks):
    """Calculate comprehensive joint angles for gait analysis"""
    angles = []
    
    # Define key points
    left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
    right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
    neck = (left_shoulder + right_shoulder) / 2
    left_hip = np.array([landmarks[23].x, landmarks[23].y])
    right_hip = np.array([landmarks[24].x, landmarks[24].y])
    left_knee = np.array([landmarks[25].x, landmarks[25].y])
    right_knee = np.array([landmarks[26].x, landmarks[26].y])
    left_ankle = np.array([landmarks[27].x, landmarks[27].y])
    right_ankle = np.array([landmarks[28].x, landmarks[28].y])
    
    # Calculate angles
    angles.append(angle_between(left_shoulder - neck, left_hip - left_shoulder))  # Left shoulder
    angles.append(angle_between(right_shoulder - neck, right_hip - right_shoulder))  # Right shoulder
    angles.append(angle_between(left_hip - left_knee, left_shoulder - left_hip))  # Left hip
    angles.append(angle_between(right_hip - right_knee, right_shoulder - right_hip))  # Right hip
    angles.append(angle_between(left_knee - left_ankle, left_hip - left_knee))  # Left knee
    angles.append(angle_between(right_knee - right_ankle, right_hip - right_knee))  # Right knee
    
    return angles

def extract_gait_features(video_path, temporal_window=30):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None

    temporal_features = []
    frame_count = 0
    prev_feats = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
            continue

        # Resize and convert frame
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 1. Basic landmark coordinates (x,y) for 25 key points
            frame_feats = []
            for i in range(25):  # Using first 25 landmarks (0-24)
                lm = landmarks[i]
                frame_feats.extend([lm.x, lm.y])
            
            # 2. Joint angles (6 angles)
            angles = calculate_advanced_joint_angles(landmarks)
            frame_feats.extend(angles)
            
            # 3. Velocity features (if available)
            if prev_feats is not None:
                velocities = np.array(frame_feats[:50]) - np.array(prev_feats[:50])  # First 25 landmarks (x,y)
                frame_feats.extend(velocities.tolist())
            
            # Ensure fixed length (50 landmarks + 6 angles + 50 velocities = 106)
            if len(frame_feats) < 106:
                frame_feats.extend([0]*(106 - len(frame_feats)))
            
            temporal_features.append(frame_feats[:106])  # Truncate if longer
            prev_feats = frame_feats[:50]  # Store current landmarks for next frame

    cap.release()
    pose.close()

    if len(temporal_features) < temporal_window:
        print(f"[ERROR] Only got {len(temporal_features)} frames (need {temporal_window})")
        return None

    # Convert to numpy array
    temporal_features = np.array(temporal_features, dtype=np.float32)
    
    # Calculate statistics
    mean_features = np.mean(temporal_features, axis=0)
    std_features = np.std(temporal_features, axis=0)
    
    # Combine features
    final_features = np.concatenate([mean_features, std_features])
    
    return final_features if not np.any(np.isnan(final_features)) else None