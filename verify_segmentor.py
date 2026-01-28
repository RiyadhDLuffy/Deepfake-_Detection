import sys
import os
import cv2
import numpy as np
from PIL import Image
import json

# Force UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.models.segmentor import get_segmentor

def verify_segmentor():
    # Path to directory
    base_dir = r"c:\Users\userw\Desktop\مهند\شلبي\Datasets\real fake images\test\REAL"
    
    # Get first 5 images
    images = [f for f in os.listdir(base_dir) if f.endswith('.jpg')][:5]
    
    segmentor = get_segmentor()
    
    for img_name in images:
        image_path = os.path.join(base_dir, img_name)
        print(f"\nTesting segmentor on {img_name}")
        
        try:
            image = Image.open(image_path)
            result = segmentor.segment(image)
            
            print(f"   Faces detected: {result['faces_detected']}")
            print(f"   Fake Percentage: {result['fake_percentage']}%")
            
            if result['faces_detected'] > 0:
                print("   ✅ Face detection working")
            else:
                print("   ⚠️ No faces detected")

        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    verify_segmentor()
