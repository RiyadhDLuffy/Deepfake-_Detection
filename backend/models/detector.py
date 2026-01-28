"""
Face & Object Detector for Deepfake Analysis
Detects objects/faces using YOLOv8 (State-of-the-art)
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from ultralytics import YOLO

class FaceDetector:
    """
    Advanced Detector for Deepfake Analysis
    Uses YOLOv8 to detect faces/objects as required by project goals
    """
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        # Load pre-trained YOLOv8 model
        self.model = YOLO(model_path)
        print(f"✅ Detector ready: YOLOv8 Mode ({model_path})")
    
    def detect(self, image: Image.Image, confidence_threshold: float = 0.25) -> dict:
        """
        Detect objects/faces in the image using YOLOv8
        """
        # Convert PIL to CV2
        img_cv = np.array(image.convert('RGB'))
        
        # Run YOLOv8 inference
        results = self.model(img_cv, conf=confidence_threshold, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                # We are primarily interested in 'person' or 'face'
                # Standard YOLOv8n has 'person' (index 0)
                detections.append({
                    "box": {
                        "x": int(x1), 
                        "y": int(y1), 
                        "width": int(x2 - x1), 
                        "height": int(y2 - y1)
                    },
                    "confidence": round(conf, 2),
                    "label": label,
                    "label_ar": "شخص" if label == "person" else label
                })
        
        # Draw boxes
        annotated_image = self._draw_boxes(image.copy(), detections)
        img_base64 = self._image_to_base64(annotated_image)
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "annotated_image": f"data:image/png;base64,{img_base64}",
            "engine": "YOLOv8"
        }
    
    def _draw_boxes(self, image: Image.Image, detections: list) -> Image.Image:
        """Draw detection boxes on image with Cyber theme"""
        draw = ImageDraw.Draw(image)
        
        for det in detections:
            box = det["box"]
            x, y, w, h = box["x"], box["y"], box["width"], box["height"]
            conf = det["confidence"]
            label = det["label"].upper()
            
            # Cyan color for Cyber theme
            color = '#00f2ff'
            
            # Draw rectangle
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            
            # Draw corners for "Tech" look
            len_line = int(w * 0.15)
            # Top Left
            draw.line([(x, y), (x + len_line, y)], fill=color, width=6)
            draw.line([(x, y), (x, y + len_line)], fill=color, width=6)
            # Top Right
            draw.line([(x+w, y), (x+w - len_line, y)], fill=color, width=6)
            draw.line([(x+w, y), (x+w, y + len_line)], fill=color, width=6)
            
            # Label background
            label_text = f"{label} {conf:.0%}"
            draw.rectangle([x, y - 20, x + 120, y], fill=color)
            draw.text((x + 5, y - 18), label_text, fill='black')
        
        return image
    
    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

# Singleton instance
_detector_instance = None

def get_detector(model_path: str = "yolov8n.pt") -> FaceDetector:
    """Get or create detector singleton"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector(model_path)
    return _detector_instance
