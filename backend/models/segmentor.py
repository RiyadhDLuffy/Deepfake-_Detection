import torch
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageDraw
import io
import base64
import cv2
import os
from torchvision import models, transforms

# Import FaceDetector
from .detector import get_detector

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepfakeSegmentor:
    """Advanced Segmentation using DeepLabV3 as required by project goals"""
    
    def __init__(self, model_path: str = None):
        # Load pre-trained DeepLabV3
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        ).to(device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(520),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize face detector
        self.detector = get_detector()
        print(f"✅ Segmentor ready: DeepLabV3 Segmentation Mode")
    
    def segment(self, image: Image.Image) -> dict:
        """Segment regions in the image using DeepLabV3"""
        original_size = image.size
        
        # 1. Preprocess and Run Model
        input_tensor = self.preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get the segmentation mask (argmax over classes)
        output_predictions = output.argmax(0).cpu().numpy()
        
        # Class 15 is 'person' in Pascal VOC (which DeepLabV3 is trained on)
        # We focus on persons for deepfake detection
        person_mask = (output_predictions == 15).astype(np.float32)
        
        # Resize mask back to original size
        person_mask_cv = cv2.resize(person_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        # 2. Forensic Analysis (ELA) - Keep this as a secondary 'Heatmap'
        ela_mask = self._generate_ela_mask(image)
        
        # 3. Combine: Show ELA heat within the segmented person
        final_mask = ela_mask * person_mask_cv
        
        mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))
        
        # Create colored overlay
        overlay = self._create_overlay(image, final_mask)
        
        # Calculate statistics
        fake_percentage = np.mean(final_mask) * 100 if np.sum(person_mask_cv) > 0 else 0.0
        
        # Convert images to base64
        mask_b64 = self._image_to_base64(mask_pil.convert('RGB'))
        overlay_b64 = self._image_to_base64(overlay)
        
        return {
            "mask_image": f"data:image/png;base64,{mask_b64}",
            "overlay_image": f"data:image/png;base64,{overlay_b64}",
            "disease_percentage": round(fake_percentage, 2),
            "fake_percentage": round(fake_percentage, 2),
            "severity": self._get_severity(fake_percentage),
            "severity_ar": self._get_severity_ar(fake_percentage),
            "engine": "DeepLabV3 + ELA"
        }
    
    def _generate_ela_mask(self, image: Image.Image) -> np.ndarray:
        """Forensic Error Level Analysis"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        buffered.seek(0)
        resaved = Image.open(buffered)
        ela_im = ImageChops.difference(image, resaved)
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
        ela_cv = np.array(ela_im.convert('L'))
        _, mask = cv2.threshold(ela_cv, 40, 255, cv2.THRESH_BINARY) # Lower threshold for more sensitivity
        return mask / 255.0
    
    def _create_overlay(self, original: Image.Image, mask: np.ndarray) -> Image.Image:
        """Create colored overlay showing manipulated regions"""
        overlay = np.array(original)
        # Red/Hot overlay for fake regions
        mask_3d = np.zeros_like(overlay)
        mask_3d[:,:,0] = (mask * 255).astype(np.uint8) # Red channel
        
        alpha = 0.5
        blended = cv2.addWeighted(overlay, 1.0, mask_3d, alpha, 0)
        return Image.fromarray(blended)
    
    def _get_severity(self, percentage: float) -> str:
        if percentage < 0.5: return "Authentic"
        elif percentage < 5: return "Low Manipulation"
        else: return "High Manipulation"
    
    def _get_severity_ar(self, percentage: float) -> str:
        if percentage < 0.5: return "أصلي"
        elif percentage < 5: return "تلاعب طفيف"
        else: return "تلاعب عالي"
    
    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

# Singleton instance
_segmentor_instance = None

def get_segmentor(model_path: str = None) -> DeepfakeSegmentor:
    """Get or create segmentor singleton"""
    global _segmentor_instance
    if _segmentor_instance is None:
        _segmentor_instance = DeepfakeSegmentor(model_path)
    return _segmentor_instance
