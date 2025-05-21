import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from extract_features import extract_gait_features
from utils import load_dataset


class GaitAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gait Authentication System")
        self.root.geometry("1200x800")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Load dataset
        self.labels, self.features_dataset, self.scaler, self.lof = load_dataset()
        
        # Video variables
        self.cap = None
        self.video_path = ""
        self.recognition_result = None
        self.after_id = None
        self.is_webcam_mode = False
        self.feature_buffer = []
        self.buffer_size = 30  # Number of frames to buffer for real-time analysis
        
        # Create UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - controls
        control_frame = tk.Frame(main_frame, width=250, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        control_frame.pack_propagate(False)
        
        # Mode selection
        mode_label = tk.Label(control_frame, text="Select Mode:", bg="#f0f0f0", font=('Helvetica', 10, 'bold'))
        mode_label.pack(pady=(10,5))
        
        self.mode_var = tk.StringVar(value="file")
        tk.Radiobutton(control_frame, text="Video File", variable=self.mode_var, 
                      value="file", bg="#f0f0f0", command=self.toggle_mode).pack(anchor=tk.W)
        tk.Radiobutton(control_frame, text="Webcam", variable=self.mode_var, 
                      value="webcam", bg="#f0f0f0", command=self.toggle_mode).pack(anchor=tk.W)
        
        # Control buttons
        btn_style = {'padx': 10, 'pady': 5, 'width': 20}
        
        self.load_btn = tk.Button(control_frame, text="Load Video", command=self.load_video, **btn_style)
        self.load_btn.pack(pady=10)
        
        self.process_btn = tk.Button(control_frame, text="Process Gait", command=self.process_gait, 
                                    state=tk.DISABLED, **btn_style)
        self.process_btn.pack(pady=10)
        
        self.play_btn = tk.Button(control_frame, text="Start Webcam", command=self.toggle_webcam, 
                                 state=tk.DISABLED, **btn_style)
        self.play_btn.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Status: Ready", bg="#f0f0f0", 
                                    font=('Helvetica', 12), wraplength=200)
        self.status_label.pack(pady=20, padx=10)
        
        # Right panel - video display
        self.video_frame = tk.Frame(main_frame, bg='black')
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(expand=True)
        
    def toggle_mode(self):
        if self.mode_var.get() == "file":
            self.load_btn.config(text="Load Video", command=self.load_video)
            self.play_btn.config(text="Play Video", command=self.play_video)
            self.is_webcam_mode = False
            self.stop_webcam()
        else:
            self.load_btn.config(text="Select Webcam", command=self.select_webcam)
            self.play_btn.config(text="Start Webcam", command=self.toggle_webcam)
            self.is_webcam_mode = True
            self.stop_video()
    
    def select_webcam(self):
        # Simple webcam selection dialog
        self.cap = cv2.VideoCapture(0)  # Default webcam
        if self.cap.isOpened():
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Webcam Selected")
            # Show first frame
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.status_label.config(text="Status: Webcam Not Found")
    
    def toggle_webcam(self):
        if self.play_btn.cget("text") == "Start Webcam":
            self.play_btn.config(text="Stop Webcam")
            self.start_webcam()
        else:
            self.play_btn.config(text="Start Webcam")
            self.stop_webcam()
    
    def start_webcam(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        self.update_webcam()
    
    def stop_webcam(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def stop_video(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def load_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")])
        
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Video Loaded\n" + self.video_path.split('/')[-1])
            
            # Show first frame
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def process_gait(self):
        if self.is_webcam_mode:
            self.status_label.config(text="Status: Ready for Real-Time Analysis")
            self.play_btn.config(state=tk.NORMAL)
        else:
            if self.video_path:
                self.status_label.config(text="Status: Processing gait features...")
                self.root.update()
                
                # Extract features and recognize
                video_features = extract_gait_features(self.video_path)
                if video_features is not None:
                    features = self.scaler.transform([video_features])
                    cos_sims = cosine_similarity(features, self.features_dataset)[0]
                    best_idx = np.argmax(cos_sims)
                    best_score = cos_sims[best_idx]
                    
                    is_inlier = self.lof.predict(features)[0] == 1
                    # threshold = np.mean(cos_sims) + 1.5 * np.std(cos_sims)
                    threshold = np.percentile(cos_sims, 95)
                    
                    if is_inlier and best_score >= threshold:
                        self.recognition_result = {
                            "name": self.labels.iloc[best_idx],
                            "score": best_score,
                            "color": (0, 255, 0)  # Green
                        }
                    else:
                        self.recognition_result = {
                            "name": "Unknown",
                            "score": best_score,
                            "color": (0, 0, 255)  # Red
                        }
                    
                    self.status_label.config(
                        text=f"Status: Recognized\n{self.recognition_result['name']}\n"
                             f"Confidence: {self.recognition_result['score']*100:.1f}%")
                    self.play_btn.config(state=tk.NORMAL)
                else:
                    self.status_label.config(text="Status: Feature extraction failed")
    
    def play_video(self):
        if self.cap is not None and self.recognition_result is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.update_video()
    
    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        
        # Process frame for visualization
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
            
            # Draw recognition results
            cv2.rectangle(frame, (50, 50), (frame.shape[1]-50, frame.shape[0]-50),
                         self.recognition_result["color"], 2)
            cv2.putText(frame, 
                       f"{self.recognition_result['name']} ({self.recognition_result['score']*100:.1f}%)",
                       (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, self.recognition_result["color"], 2)
        
        self.display_frame(frame)
        self.after_id = self.root.after(30, self.update_video)
    
    def update_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Process frame for visualization
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
            
            # Add current frame to feature buffer
            self.feature_buffer.append(frame)
            if len(self.feature_buffer) > self.buffer_size:
                self.feature_buffer.pop(0)
            
            # Perform real-time recognition every buffer_size frames
            if len(self.feature_buffer) == self.buffer_size and len(self.feature_buffer) % 10 == 0:
                try:
                    video_features = extract_gait_features_from_frames(self.feature_buffer)
                    if video_features is not None:
                        features = self.scaler.transform([video_features])
                        cos_sims = cosine_similarity(features, self.features_dataset)[0]
                        best_idx = np.argmax(cos_sims)
                        best_score = cos_sims[best_idx]
                        
                        is_inlier = self.lof.predict(features)[0] == 1
                        threshold = np.mean(cos_sims) + 1.5 * np.std(cos_sims)
                        
                        if is_inlier and best_score >= threshold:
                            self.recognition_result = {
                                "name": self.labels.iloc[best_idx],
                                "score": best_score,
                                "color": (0, 255, 0)  # Green
                            }
                        else:
                            self.recognition_result = {
                                "name": "Unknown",
                                "score": best_score,
                                "color": (0, 0, 255)  # Red
                            }
                except Exception as e:
                    print(f"Real-time processing error: {e}")
        
        # Draw recognition results if available
        if self.recognition_result:
            cv2.rectangle(frame, (50, 50), (frame.shape[1]-50, frame.shape[0]-50),
                         self.recognition_result["color"], 2)
            cv2.putText(frame, 
                       f"{self.recognition_result['name']} ({self.recognition_result['score']*100:.1f}%)",
                       (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, self.recognition_result["color"], 2)
        
        self.display_frame(frame)
        self.after_id = self.root.after(30, self.update_webcam)
    
    def display_frame(self, frame):
        # Convert to PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # Resize while maintaining aspect ratio
        max_width = self.video_frame.winfo_width() - 20
        max_height = self.video_frame.winfo_height() - 20
        
        width_ratio = max_width / img.width
        height_ratio = max_height / img.height
        ratio = min(width_ratio, height_ratio)
        
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
    
    def on_closing(self):
        self.stop_webcam()
        self.stop_video()
        if hasattr(self, 'pose'):
            self.pose.close()
        self.root.destroy()

# Helper function for real-time feature extraction
def extract_gait_features_from_frames(frames):
    # Implement your frame-based feature extraction here
    # This should process multiple frames and return features
    # Similar to extract_gait_features but for frames list
    
    # Placeholder implementation - replace with your actual frame processing
    try:
        # Process each frame and aggregate features
        all_features = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Add your feature extraction logic here
            # This is just a placeholder
            all_features.append(np.random.rand(100))  # Replace with real features
        
        # Average features across frames
        return np.mean(all_features, axis=0)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = GaitAuthApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()