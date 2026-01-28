"""
Deepfake Classifier using Transfer Learning with MobileNetV2
PyTorch Version - Handles Real vs Fake image classification
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
import os
import json

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deepfake Classes
DISEASE_CLASSES = [
    'FAKE', 'REAL'
]

# Arabic translations for Deepfake classes
DISEASE_NAMES_AR = {
    'FAKE': 'صورة مزيفة (AI Generated)',
    'REAL': 'صورة حقيقية (Real)'
}


class DeepfakeClassifier:
    """Deepfake classification using MobileNetV2 with Transfer Learning (PyTorch)"""
    
    def __init__(self, model_path: str = None):
        self.img_size = (224, 224)
        self.num_classes = len(DISEASE_CLASSES)
        self.model = None
        self.model_path = model_path
        
        # Image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build the classification model using MobileNetV2 as base"""
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Match the architecture in train_deepfake.py
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, self.num_classes)
        )
        
        self.model = self.model.to(device)
        self.model.eval()
        print(f"✅ Classifier model built with {self.num_classes} classes (PyTorch on {device})")
    
    def load_model(self, path: str):
        """Load a trained model from disk"""
        try:
            self.build_model()
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"✅ Model loaded from {path}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}. Using pretrained model...")
            self.build_model()
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0)
        return img_tensor.to(device)
    
    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        """
        Predict Real vs Fake for an image
        """
        if self.model is None:
            self.build_model()
        
        self.model.eval()
        
        # Get Model Prediction
        processed_img = self.preprocess_image(image)
        outputs = self.model(processed_img)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predictions = probabilities.cpu().numpy()
        
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx])
        
        class_name = DISEASE_CLASSES[top_idx]
        class_name_ar = DISEASE_NAMES_AR.get(class_name, class_name)
        is_real = class_name == 'REAL'
        
        # Get top predictions (all of them since only 2)
        top_indices = np.argsort(predictions)[::-1]
        top_predictions = [
            {
                "class_name": DISEASE_CLASSES[idx],
                "class_name_ar": DISEASE_NAMES_AR.get(DISEASE_CLASSES[idx], DISEASE_CLASSES[idx]),
                "confidence": float(predictions[idx])
            }
            for idx in top_indices
        ]
        
        return {
            "class_name": class_name,
            "class_name_ar": class_name_ar,
            "confidence": confidence,
            "is_healthy": is_real, # Legacy field for compatibility
            "is_real": is_real,
            "is_fake": not is_real,
            "top_5_predictions": top_predictions
        }
    
    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """Predict from raw image bytes"""
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict(image)


# Singleton instance
_classifier_instance = None

def get_classifier(model_path: str = None) -> DeepfakeClassifier:
    """Get or create the classifier singleton"""
    global _classifier_instance
    if _classifier_instance is None:
        if model_path is None:
            default_path = os.path.join(os.path.dirname(__file__), "deepfake_model.pth")
            if os.path.exists(default_path):
                model_path = default_path
        _classifier_instance = DeepfakeClassifier(model_path)
    return _classifier_instance
