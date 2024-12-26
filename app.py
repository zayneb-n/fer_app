import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="🎭 Facial Emotion Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

class EmotionRecognizer:
    def __init__(self):
        self.emotion_dict = {
            0: "Angry", 1: "Disgusted", 2: "Fearful",
            3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
        }
        self.model_path = Path('models/final_model_20241221_040330.keras')
        self.model = None
        self.face_detection = None
        
    def load_model(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            self.model = load_model(self.model_path)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
            
    def initialize_face_detection(self):
        try:
            mp_face_detection = mp.solutions.face_detection
            self.face_detection = mp_face_detection.FaceDetection(
                min_detection_confidence=0.5,
                model_selection=0
            )
            return True
        except Exception as e:
            logger.error(f"Error initializing face detection: {str(e)}")
            return False

    def process_frame(self, frame):
        if frame is None:
            return None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                frame = self._process_detection(frame, detection)
        
        return frame
    
    def _process_detection(self, frame, detection):
        ih, iw, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box
        x, y, w, h = [int(coord * dim) for coord, dim in zip(
            [bbox.xmin, bbox.ymin, bbox.width, bbox.height],
            [iw, ih, iw, ih]
        )]
        
        x, y = max(0, x), max(0, y)
        w = min(w, iw - x)
        h = min(h, ih - y)
        
        try:
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0:
                return frame
                
            emotion_label, confidence = self._predict_emotion(face_region)
            
            label = f"{emotion_label}: {confidence:.1f}%"
            cv2.putText(frame, label, (x, max(y-10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        except Exception as e:
            logger.error(f"Error processing detection: {str(e)}")
            
        return frame
    
    def _predict_emotion(self, face_region):
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (48, 48))
        normalized_face = resized_face / 255.0
        preprocessed_face = np.expand_dims(np.expand_dims(normalized_face, -1), 0)
        
        prediction = self.model.predict(preprocessed_face, verbose=0)
        emotion_index = np.argmax(prediction)
        
        return (
            self.emotion_dict[emotion_index],
            float(prediction[0][emotion_index] * 100)
        )

def main():
    st.title("🎭 Facial Emotion Recognition")
    
    # Initialize session states
    if 'recognizer' not in st.session_state:
        st.session_state['recognizer'] = EmotionRecognizer()
    if 'camera_running' not in st.session_state:
        st.session_state['camera_running'] = False
    if 'camera' not in st.session_state:
        st.session_state['camera'] = None
    
    # Sidebar with app information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        ### Developer
        **Salma Feryel Jamli**
        
        ### Contact
        📧 salma.feryel21@gmail.com
        
        ### Version
        🔖 1.0.0 (December 2024)
        
        ### Features
        - Real-time emotion detection
        - 7 emotion classifications
        - Face detection with bounding box
        - Confidence score display
        
        ### Technologies Used
        - TensorFlow
        - OpenCV
        - MediaPipe
        - Streamlit
        """)
    
    # Camera controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera"):
            start_camera()
    
    with col2:
        if st.button("Stop Camera"):
            stop_camera()
    
    # Camera feed placeholder
    camera_placeholder = st.empty()
    
    # Main camera loop
    if st.session_state['camera_running'] and st.session_state['camera'] is not None:
        try:
            ret, frame = st.session_state['camera'].read()
            if ret:
                processed_frame = st.session_state['recognizer'].process_frame(frame)
                if processed_frame is not None:
                    camera_placeholder.image(processed_frame, channels="BGR")
        except Exception as e:
            logger.error(f"Error in camera feed: {str(e)}")
            stop_camera()
    else:
        st.info("Camera is stopped. Click 'Start Camera' to begin.")

def start_camera():
    if not st.session_state['camera_running']:
        if not st.session_state['recognizer'].load_model():
            st.error("Failed to load emotion recognition model")
            return
        if not st.session_state['recognizer'].initialize_face_detection():
            st.error("Failed to initialize face detection")
            return
        
        st.session_state['camera'] = cv2.VideoCapture(0)
        if not st.session_state['camera'].isOpened():
            st.error("Failed to open camera")
            return
            
        st.session_state['camera_running'] = True

def stop_camera():
    st.session_state['camera_running'] = False
    if st.session_state['camera'] is not None:
        st.session_state['camera'].release()
        st.session_state['camera'] = None

if __name__ == "__main__":
    main()