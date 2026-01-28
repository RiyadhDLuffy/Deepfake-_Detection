import sys
import os
from PIL import Image
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.classifier import get_classifier
from models.detector import get_detector

def test_models():
    print("🚀 Testing Deepfake Models...")
    
    # Create dummy image (black square)
    img = Image.new('RGB', (224, 224), color='black')
    
    # Test Classifier
    print("\n1️⃣ Testing Classifier...")
    classifier = get_classifier()
    result = classifier.predict(img)
    print("Result:", result)
    
    assert "class_name" in result
    assert "confidence" in result
    assert result["class_name"] in ["REAL", "FAKE"]
    print("✅ Classifier Test Passed")
    
    # Test Detector
    print("\n2️⃣ Testing Detector...")
    detector = get_detector()
    det_result = detector.detect(img)
    print("Result:", det_result)
    
    assert "detections" in det_result
    assert "annotated_image" in det_result
    print("✅ Detector Test Passed")

if __name__ == "__main__":
    test_models()
