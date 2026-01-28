"""
Image Comparator for Deepfake Analysis
Compares original vs potentially manipulated images
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

class DeepfakeComparator:
    """
    Compare two images to detect manipulation or changes
    """
    
    def __init__(self):
        pass
    
    def compare(self, image1: Image.Image, image2: Image.Image) -> Dict:
        """
        Full comparison between two images
        image1: Reference/Original
        image2: Suspect/Manipulated
        """
        # Convert to numpy arrays
        img1 = np.array(image1.convert('RGB'))
        img2 = np.array(image2.convert('RGB'))
        
        # Resize to same dimensions if needed
        if img1.shape != img2.shape:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        # Convert to BGR for OpenCV
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        # Calculate metrics
        similarity = self._calculate_similarity(img1_bgr, img2_bgr)
        diff_image = self._create_diff_image(img1, img2)
        side_by_side = self._create_side_by_side(img1, img2)
        
        return {
            "similarity": similarity,
            "diff_image": diff_image,
            "side_by_side": side_by_side,
            "summary": self._generate_summary(similarity)
        }
    
    def _calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """Calculate structural similarity"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        if HAS_SKIMAGE:
            try:
                score, _ = ssim(gray1, gray2, full=True)
            except:
                score = self._simple_similarity(gray1, gray2)
        else:
            score = self._simple_similarity(gray1, gray2)
            
        return {
            "ssim_score": float(round(score * 100, 1)),
            "is_identical": score > 0.99,
            "analysis_ar": f"نسبة التطابق: {round(score * 100)}%"
        }
    
    def _simple_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simple similarity fallback"""
        diff = np.abs(img1.astype(float) - img2.astype(float))
        return 1.0 - (np.mean(diff) / 255.0)
    
    def _create_diff_image(self, img1: np.ndarray, img2: np.ndarray) -> str:
        """Create visual difference image"""
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        diff_colored = np.zeros_like(img1)
        diff_colored[thresh > 0] = [255, 0, 0] # Red for differences
        
        alpha = 0.5
        overlay = cv2.addWeighted(img1, 1 - alpha, diff_colored, alpha, 0)
        
        pil_img = Image.fromarray(overlay)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    
    def _create_side_by_side(self, img1: np.ndarray, img2: np.ndarray) -> str:
        """Create side-by-side comparison"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        max_h = max(h1, h2)
        if h1 != max_h:
            scale = max_h / h1
            img1 = cv2.resize(img1, (int(w1 * scale), max_h))
        if h2 != max_h:
            scale = max_h / h2
            img2 = cv2.resize(img2, (int(w2 * scale), max_h))
            
        h, w1 = img1.shape[:2]
        _, w2 = img2.shape[:2]
        
        gap = 20
        canvas = np.ones((h, w1 + gap + w2, 3), dtype=np.uint8) * 20
        canvas[:, 0:w1] = img1
        canvas[:, w1 + gap:w1 + gap + w2] = img2
        
        pil_img = Image.fromarray(canvas)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

    def _generate_summary(self, similarity: Dict) -> Dict:
        """Generate text summary"""
        score = similarity['ssim_score']
        if score > 99:
            text = "الصورتان متطابقتان تماماً"
        elif score > 90:
            text = "تغييرات طفيفة جداً"
        elif score > 50:
            text = "تغييرات ملحوظة (تلاعب محتمل)"
        else:
            text = "اختلاف جذري بين الصورتين"
            
        return {
            "summary_ar": text,
            "score": score
        }

# Singleton instance
_comparator_instance = None

def get_comparator() -> DeepfakeComparator:
    """Get or create comparator singleton"""
    global _comparator_instance
    if _comparator_instance is None:
        _comparator_instance = DeepfakeComparator()
    return _comparator_instance
