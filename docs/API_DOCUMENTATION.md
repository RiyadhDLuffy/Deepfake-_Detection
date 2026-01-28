# 🌿 Smart Plant Disease Detection API Documentation

## نظرة عامة
REST API لنظام كشف أمراض النباتات باستخدام الذكاء الاصطناعي.

**Base URL:** `http://localhost:8000`

---

## 🔐 المصادقة
لا يتطلب هذا الإصدار مصادقة.

---

## 📍 Endpoints

### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
    "status": "healthy",
    "message": "API is running",
    "models": {
        "classifier": "ready",
        "detector": "ready",
        "segmentor": "ready",
        "video_processor": "ready"
    }
}
```

---

### 2. Get Disease Classes
```http
GET /api/classes
```

**Response:**
```json
{
    "total_classes": 38,
    "classes": [
        {
            "id": 0,
            "name": "Apple___Apple_scab",
            "name_ar": "جرب التفاح",
            "is_healthy": false
        },
        ...
    ]
}
```

---

### 3. Image Classification
```http
POST /api/classify
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | ✅ | صورة النبات (JPEG, PNG) |

**Response:**
```json
{
    "success": true,
    "result": {
        "class_name": "Tomato___Early_blight",
        "class_name_ar": "اللفحة المبكرة للطماطم",
        "confidence": 0.95,
        "is_healthy": false,
        "top_5_predictions": [
            {
                "class_name": "Tomato___Early_blight",
                "class_name_ar": "اللفحة المبكرة للطماطم",
                "confidence": 0.95
            },
            ...
        ]
    }
}
```

---

### 4. Object Detection
```http
POST /api/detect
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | ✅ | صورة النبات |
| confidence_threshold | float | ❌ | حد الثقة (0-1)، الافتراضي: 0.3 |

**Response:**
```json
{
    "success": true,
    "result": {
        "detections": [
            {
                "box": {
                    "x": 100,
                    "y": 150,
                    "width": 200,
                    "height": 180
                },
                "confidence": 0.87,
                "label": "disease_region",
                "label_ar": "منطقة مصابة"
            }
        ],
        "num_detections": 1,
        "annotated_image": "data:image/png;base64,..."
    }
}
```

---

### 5. Image Segmentation
```http
POST /api/segment
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | ✅ | صورة النبات |

**Response:**
```json
{
    "success": true,
    "result": {
        "mask_image": "data:image/png;base64,...",
        "overlay_image": "data:image/png;base64,...",
        "disease_percentage": 15.5,
        "severity": "Mild",
        "severity_ar": "إصابة خفيفة"
    }
}
```

---

### 6. Full Analysis
```http
POST /api/analyze
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | ✅ | صورة النبات |

**Response:**
```json
{
    "success": true,
    "result": {
        "classification": { ... },
        "detection": { ... },
        "segmentation": { ... }
    }
}
```

---

### 7. Video Frame Processing
```http
POST /api/video-frame
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | ✅ | إطار فيديو (JPEG) |

---

### 8. Real-time Video Stream (WebSocket)
```
WS /api/video-stream
```

**Send:** Binary frame data (JPEG)
**Receive:** JSON analysis results

---

## 📊 رموز الاستجابة

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 500 | Internal Server Error |

---

## 🛠️ أمثلة الاستخدام

### Python
```python
import requests

# تصنيف صورة
with open('plant.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/classify',
        files={'file': f}
    )
    print(response.json())
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/classify" \
  -F "file=@plant.jpg"
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('/api/classify', {
    method: 'POST',
    body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 📚 Swagger Documentation

الوصول إلى توثيق Swagger التفاعلي:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
