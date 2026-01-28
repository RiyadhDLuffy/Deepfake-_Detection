"""
Real-time Video Processing for Deepfake Detection
Handles webcam streams and video file analysis
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io
import time
from typing import Generator, AsyncGenerator

# Import our models
from .classifier import DeepfakeClassifier, get_classifier
from .detector import FaceDetector, get_detector

class DeepfakeVideoProcessor:
    """
    Real-time video processing for deepfake detection
    """
    
    def __init__(self):
        self.classifier = None
        self.detector = None
        self.is_processing = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
    def initialize_models(self):
        """Initialize AI models (lazy loading)"""
        if self.classifier is None:
            self.classifier = get_classifier()
        if self.detector is None:
            self.detector = get_detector()
    
    def process_frame(self, frame: np.ndarray, return_frame: bool = True) -> dict:
        """Process a single video frame"""
        self.initialize_models()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Classification
        classification = self.classifier.predict(pil_image)
        
        # Detection
        detection_result = self.detector.detect(pil_image, confidence_threshold=0.7)
        
        # Logic for severity/status
        is_fake = classification['class_name'] == 'FAKE'
        status_ar = "مزيف" if is_fake else "حقيقي"
        
        results = {
            "classification": classification,
            "detections": detection_result['detections'],
            "status_ar": status_ar,
            "fps": self.fps
        }

        if return_frame:
            annotated_frame = self._annotate_frame(
                frame.copy(), 
                classification, 
                detection_result['detections']
            )
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            results["frame"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
            
        return results
    
    def _annotate_frame(self, frame: np.ndarray, classification: dict, 
                        detections: list) -> np.ndarray:
        """Draw annotations on frame"""
        h, w = frame.shape[:2]
        
        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text
        is_fake = classification['class_name'] == 'FAKE'
        color = (0, 0, 255) if is_fake else (0, 255, 0)
        status_text = "FAKE" if is_fake else "REAL"
        conf_text = f"{classification['confidence']:.1%}"
        
        cv2.putText(frame, f"Status: {status_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Conf: {conf_text}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps}", (w - 120, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw Faces
        for det in detections:
            box = det['box']
            x, y, bw, bh = box['x'], box['y'], box['width'], box['height']
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return frame
    
    def process_webcam_frame(self, frame_data: bytes) -> dict:
        """Process webcam frame from bytes"""
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return {"error": "Decode failed"}
        return self.process_frame(frame, return_frame=False)

# Singleton instance
_video_processor_instance = None

def get_video_processor() -> DeepfakeVideoProcessor:
    """Get or create video processor singleton"""
    global _video_processor_instance
    if _video_processor_instance is None:
        _video_processor_instance = DeepfakeVideoProcessor()
    return _video_processor_instance
