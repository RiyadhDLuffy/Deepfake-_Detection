"""
Advanced Image Analyzer for Deepfake Detection
Provides forensic analysis (ELA, Noise, Metadata)
"""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import base64
from typing import Dict

class DeepfakeAnalyzer:
    """
    Advanced image analysis for deepfake detection
    Features: ELA, Noise Analysis, Frequency Analysis
    """
    
    def __init__(self):
        pass
    
    def full_analysis(self, image: Image.Image) -> Dict:
        """Perform complete advanced forensic analysis"""
        img_cv = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        return {
            "ela_analysis": self.analyze_ela(image),
            "noise_analysis": self.analyze_noise(img_cv),
            "frequency_analysis": self.analyze_frequency(img_cv),
            "metadata_analysis": self.analyze_metadata(image),
            "quality_metrics": self.calculate_quality_metrics(img_cv)
        }
    
    def analyze_ela(self, image: Image.Image) -> Dict:
        """Error Level Analysis"""
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
        
        # Convert to base64
        buffered_ela = io.BytesIO()
        ela_im.save(buffered_ela, format="PNG")
        ela_base64 = base64.b64encode(buffered_ela.getvalue()).decode()
        
        return {
            "ela_image": f"data:image/png;base64,{ela_base64}",
            "max_difference": max_diff,
            "analysis_ar": "تحليل مستوى الخطأ (ELA) يكشف عن مناطق التلاعب في الضغط."
        }
    
    def analyze_noise(self, img_cv: np.ndarray) -> Dict:
        """Analyze noise patterns"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Simple noise estimation using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_variance = np.var(laplacian)
        
        return {
            "noise_variance": round(noise_variance, 2),
            "consistency": "Consistent" if noise_variance < 1000 else "Inconsistent (Suspicious)",
            "analysis_ar": f"تباين الضوضاء: {round(noise_variance)}"
        }
    
    def analyze_frequency(self, img_cv: np.ndarray) -> Dict:
        """Frequency Domain Analysis (FFT)"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Normalize for display
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert to base64
        pil_img = Image.fromarray(magnitude_spectrum)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        fft_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "fft_image": f"data:image/png;base64,{fft_base64}",
            "analysis_ar": "تحليل التردد يكشف عن الأنماط المتكررة الناتجة عن التوليد الآلي (GANs)."
        }
    
    def analyze_metadata(self, image: Image.Image) -> Dict:
        """Extract basic metadata"""
        info = image.info
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "exif_present": "exif" in info,
            "analysis_ar": "تحليل البيانات الوصفية للصورة."
        }
    
    def calculate_quality_metrics(self, img_cv: np.ndarray) -> Dict:
        """Calculate image quality metrics"""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            "blur_score": round(blur_score, 2),
            "sharpness": "Sharp" if blur_score > 100 else "Blurry",
            "analysis_ar": f"درجة الحدة: {round(blur_score)}"
        }
    
    def enhance_image(self, image: Image.Image, enhancement_type: str = "auto") -> Dict:
        """Enhance image for better analysis"""
        img_cv = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        if enhancement_type == "auto":
            enhanced = cv2.detailEnhance(img_cv, sigma_s=10, sigma_r=0.15)
        elif enhancement_type == "brightness":
            enhanced = cv2.convertScaleAbs(img_cv, alpha=1.0, beta=50)
        elif enhancement_type == "contrast":
            enhanced = cv2.convertScaleAbs(img_cv, alpha=1.5, beta=0)
        elif enhancement_type == "sharpen":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(img_cv, -1, kernel)
        else:
            enhanced = img_cv
            
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(enhanced_rgb)
        
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "enhanced_image": f"data:image/png;base64,{enhanced_base64}",
            "enhancement_type": enhancement_type
        }

# Singleton instance
_analyzer_instance = None

def get_analyzer() -> DeepfakeAnalyzer:
    """Get or create analyzer singleton"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DeepfakeAnalyzer()
    return _analyzer_instance
