"""
TruthLens Deepfake Detection API
FastAPI backend for Deepfake Analysis
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn
import os
import sys
import json
from PIL import Image
import io
import traceback
from datetime import datetime

# Add models to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Deepfake Models
from models.classifier import get_classifier
from models.detector import get_detector
from models.segmentor import get_segmentor
from models.video_processor import get_video_processor
from models.image_analyzer import get_analyzer
from models.image_comparator import get_comparator

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Pre-initializing TruthLens AI models...")
    try:
        get_classifier() 
        get_detector()   
        get_segmentor()  
        get_video_processor().initialize_models()
        get_analyzer()
        get_comparator()
        print("✅ All TruthLens models loaded")
    except Exception as e:
        print(f"⚠️ Error during model pre-initialization: {e}")
    yield
    print("👋 Shutting down...")

app = FastAPI(
    lifespan=lifespan,
    title="👁️ TruthLens Deepfake Detection API",
    description="Deepfake Detection and Forensic Analysis API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/css", StaticFiles(directory=os.path.join(frontend_path, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(frontend_path, "js")), name="js")
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(frontend_path, "index.html"))

# History Helper
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'scan_history.json')

def update_scan_history(class_name, is_real):
    try:
        if not os.path.exists(HISTORY_FILE):
             data = {"total_analyses": 0, "history": [], "class_counts": {}, "daily_stats": {}}
        else:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        data["total_analyses"] += 1
        data["class_counts"][class_name] = data["class_counts"].get(class_name, 0) + 1
        today = datetime.now().strftime('%Y-%m-%d')
        data["daily_stats"][today] = data["daily_stats"].get(today, 0) + 1
        
        data["history"].append({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "class_name": class_name,
            "is_real": is_real
        })
        if len(data["history"]) > 100:
            data["history"].pop(0)
            
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Error updating history: {e}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "system": "TruthLens AI"}

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        classifier = get_classifier()
        result = classifier.predict(image)
        
        update_scan_history(result["class_name"], result["is_real"])
            
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        detector = get_detector()
        result = detector.detect(image)
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        segmentor = get_segmentor()
        result = segmentor.segment(image)
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analyze")
async def full_analysis(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        classifier = get_classifier()
        detector = get_detector()
        segmentor = get_segmentor()
        
        classification = classifier.predict(image)
        detection = detector.detect(image)
        segmentation = segmentor.segment(image)
        
        update_scan_history(classification["class_name"], classification["is_real"])
        
        return JSONResponse(content={
            "success": True,
            "result": {
                "classification": classification,
                "detection": detection,
                "segmentation": segmentation
            }
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analyze-advanced")
async def analyze_advanced(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        analyzer = get_analyzer()
        result = analyzer.full_analysis(image)
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        img1 = Image.open(io.BytesIO(await file1.read()))
        img2 = Image.open(io.BytesIO(await file2.read()))
        comparator = get_comparator()
        result = comparator.compare(img1, img2)
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze an uploaded video file"""
    import tempfile
    import cv2
    
    temp_path = None
    try:
        # Save uploaded file to a temporary location
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
            
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Sample frames (max 10 frames to keep it fast)
        sample_rate = max(1, frame_count // 10)
        
        results = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % sample_rate == 0:
                # Process this frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                classifier = get_classifier()
                classification = classifier.predict(pil_img)
                results.append(classification)
                
            count += 1
            if len(results) >= 10: break
            
        cap.release()
        
        # Aggregate results
        if not results:
            return JSONResponse(content={"success": False, "error": "No frames processed"})
            
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        fake_votes = sum(1 for r in results if r['class_name'] == 'FAKE')
        is_fake = fake_votes > (len(results) / 2)
        
        summary = {
            "is_fake": is_fake,
            "confidence": avg_confidence,
            "status_ar": "مزيف" if is_fake else "حقيقي",
            "frames_analyzed": len(results),
            "duration_seconds": round(duration, 2)
        }
        
        return JSONResponse(content={"success": True, "result": summary})
        
    except Exception as e:
        print(f"Error analyzing video: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/statistics")
async def get_stats():
    try:
        if not os.path.exists(HISTORY_FILE):
             return {"success": True, "result": {"total_analyses": 0, "top_diseases": [], "daily_growth": []}}
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Format for frontend
        top_classes = []
        for name, count in data["class_counts"].items():
            top_classes.append({"name": name, "count": count})
            
        return {
            "success": True,
            "result": {
                "total_analyses": data["total_analyses"],
                "top_diseases": top_classes, # Keeping key name for frontend compat
                "daily_growth": list(data["daily_stats"].values()),
                "days": list(data["daily_stats"].keys())
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.websocket("/api/video-stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    processor = get_video_processor()
    try:
        while True:
            data = await websocket.receive_bytes()
            result = processor.process_webcam_frame(data)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try: await websocket.close() 
        except: pass

@app.post("/api/chat")
async def chat_with_advisor(message: dict):
    user_msg = message.get("message", "").lower()
    responses = {
        "تزييف": "لكشف التزييف، ابحث عن تشوهات في الخلفية، عدم تناسق في الإضاءة، أو ملامح وجه غير طبيعية.",
        "وقاية": "للحماية، قلل من نشر صور عالية الدقة لوجهك واستخدم أدوات توقيع الصور.",
        "أدوات": "نستخدم تقنيات CNN و ELA لكشف الآثار غير المرئية للتلاعب.",
        "دقة": "تصل دقة نظام TruthLens إلى 95% في كشف Deepfakes.",
    }
    response = "أهلاً بك في TruthLens. اسألني عن كشف التزييف أو الأدوات المستخدمة."
    for key, val in responses.items():
        if key in user_msg:
            response = val
            break
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
