# 👁️ TruthLens Deepfake Detection API Documentation

## نظرة عامة (Overview)
REST API لنظام كشف التزييف العميق (Deepfake Detection) باستخدام تقنيات الذكاء الاصطناعي والتحليل الجنائي للصور (Forensic Analysis).

**Base URL:** `http://localhost:8000`

---

## 🔐 المصادقة (Authentication)
لا يتطلب هذا الإصدار مصادقة حالياً للإستخدام المحلي.

---

## 📍 Endpoints

### 1. Health Check
التحقق من حالة النظام والموديلات.
```http
GET /api/health
```

**Response:**
```json
{
    "status": "healthy",
    "system": "TruthLens AI"
}
```

---

### 2. Image Classification (Real vs Fake)
تصنيف الصورة حقيقية أم مزيفة.
```http
POST /api/classify
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | ✅ | الصورة المراد فحصها (JPEG, PNG) |

**Response:**
```json
{
    "success": true,
    "result": {
        "class_name": "FAKE",
        "class_name_ar": "صورة مزيفة (AI Generated)",
        "confidence": 0.98,
        "is_real": false,
        "is_fake": true,
        "top_5_predictions": [...]
    }
}
```

---

### 3. Face & Object Detection (YOLOv8)
كشف الوجوه والأشخاص في الصورة وتحديد مناطق الاهتمام.
```http
POST /api/detect
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | ✅ | الصورة المرad فحصها |

**Response:**
```json
{
    "success": true,
    "result": {
        "detections": [
            {
                "box": {"x": 100, "y": 80, "width": 150, "height": 200},
                "confidence": 0.92,
                "label": "person",
                "label_ar": "شخص"
            }
        ],
        "num_detections": 1,
        "annotated_image": "data:image/png;base64,...",
        "engine": "YOLOv8"
    }
}
```

---

### 4. Forensic Segmentation (ELA)
تحليل مستوى الخطأ (Error Level Analysis) لتحديد مناطق التلاعب الرقمي.
```http
POST /api/segment
Content-Type: multipart/form-data
```

**Response:**
```json
{
    "success": true,
    "result": {
        "mask_image": "data:image/png;base64,...",
        "overlay_image": "data:image/png;base64,...",
        "fake_percentage": 15.5,
        "severity_ar": "تلاعب عالي",
        "engine": "DeepLabV3 + ELA"
    }
}
```

---

### 5. Advanced Forensic Analysis
تحليل متطور يشمل FFT (التحليل الترددي) و Noise Analysis.
```http
POST /api/analyze-advanced
Content-Type: multipart/form-data
```

---

### 6. Video Analysis
تحليل ملف فيديو عبر أخذ عينات من الإطارات.
```http
POST /api/analyze-video
Content-Type: multipart/form-data
```

---

### 7. Real-time Video Stream (WebSocket)
كشف مباشر عبر الكاميرا.
```
WS /api/video-stream
```

---

## 📊 رموز الاستجابة (Response Codes)

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 500 | Internal Server Error |

---

## 🛠️ أمثلة الاستخدام (Usage Examples)

### Python
```python
import requests

with open('face.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/classify',
        files={'file': f}
    )
    print(response.json())
```

### Script JavaScript (Frontend)
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/api/classify', {
    method: 'POST',
    body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 📚 API Visualization
الوصول إلى توثيق Swagger التفاعلي:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
