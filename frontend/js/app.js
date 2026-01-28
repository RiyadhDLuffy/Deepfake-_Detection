/**
 * TruthLens Deepfake Detection System
 * Frontend JavaScript Application
 * Version 2.0
 */
console.log('🚀 TruthLens App v2.0: System Initializing...');

const API_BASE_URL = window.location.origin;
const WS_URL = (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + window.location.host + '/api/video-stream';

// DOM Elements
const elements = {
    loadingScreen: document.getElementById('loading-screen'),
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    previewArea: document.getElementById('previewArea'),
    previewImage: document.getElementById('previewImage'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    resultsContainer: document.getElementById('resultsContainer'),

    // Results
    classificationResult: document.getElementById('classificationResult'),
    detectionResult: document.getElementById('detectionResult'),
    segmentationResult: document.getElementById('segmentationResult'),
    advancedAnalysisResult: document.getElementById('advancedAnalysisResult'),

    diseaseName: document.getElementById('diseaseName'),
    diseaseNameAr: document.getElementById('diseaseNameAr'),
    healthStatus: document.getElementById('healthStatus'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceFill: document.getElementById('confidenceFill'),

    numDetections: document.getElementById('numDetections'),
    detectionImage: document.getElementById('detectionImage'),

    diseasePercentage: document.getElementById('diseasePercentage'),
    maskImage: document.getElementById('maskImage'),
    overlayImage: document.getElementById('overlayImage'),

    // Advanced
    noiseVar: document.getElementById('noiseVar'),
    blurScore: document.getElementById('blurScore'),
    fftImage: document.getElementById('fftImage'),
    elaImage: document.getElementById('elaImage'),

    // Camera
    cameraVideo: document.getElementById('cameraVideo'),
    cameraCanvas: document.getElementById('cameraCanvas'),
    startCamera: document.getElementById('startCamera'),
    stopCamera: document.getElementById('stopCamera'),
    liveResults: document.getElementById('liveResults'),
    liveClassName: document.getElementById('liveClassName'),
    liveConfidence: document.getElementById('liveConfidence'),
    fpsValue: document.getElementById('fpsValue'),

    // Compare
    compareBox1: document.getElementById('compareBox1'),
    compareBox2: document.getElementById('compareBox2'),
    compareImg1: document.getElementById('compareImg1'),
    compareImg2: document.getElementById('compareImg2'),
    compareInput1: document.getElementById('compareInput1'),
    compareInput2: document.getElementById('compareInput2'),
    startComparison: document.getElementById('startComparison'),
    comparisonResults: document.getElementById('comparisonResults'),
    compHeadline: document.getElementById('compHeadline'),
    diffImg: document.getElementById('diffImg'),

    // Chat
    toggleChat: document.getElementById('toggleChat'),
    chatbotWidget: document.getElementById('chatbotWidget'),
    closeChat: document.getElementById('closeChat'),
    chatInput: document.getElementById('chatInput'),
    sendChat: document.getElementById('sendChat'),
    chatMessages: document.getElementById('chatMessages'),

    toast: document.getElementById('toast')
};

let state = {
    currentFile: null,
    isProcessingVideo: false,
    websocket: null,
    stream: null
};

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => elements.loadingScreen.classList.add('hidden'), 1000);
    initUpload();
    initCamera();
    initComparison();
    initChat();
});

// Upload Logic
function initUpload() {
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

    elements.analyzeBtn.addEventListener('click', async () => {
        if (!state.currentFile) return;

        elements.analyzeBtn.disabled = true;
        elements.analyzeBtn.innerHTML = '⏳ جاري التحليل...';

        try {
            const formData = new FormData();
            formData.append('file', state.currentFile);

            if (state.currentFile.type.startsWith('video/')) {
                // Video Analysis
                const res = await fetch(`${API_BASE_URL}/api/analyze-video`, {
                    method: 'POST',
                    body: formData
                }).then(r => r.json());

                if (res.success) {
                    displayVideoResults(res.result);
                } else {
                    showToast('فشل تحليل الفيديو', 'error');
                }
            } else {
                // Image Analysis (Parallel requests)
                const [clsRes, detRes, segRes, advRes] = await Promise.all([
                    fetch(`${API_BASE_URL}/api/classify`, { method: 'POST', body: formData }).then(r => r.json()),
                    fetch(`${API_BASE_URL}/api/detect`, { method: 'POST', body: formData }).then(r => r.json()),
                    fetch(`${API_BASE_URL}/api/segment`, { method: 'POST', body: formData }).then(r => r.json()),
                    fetch(`${API_BASE_URL}/api/analyze-advanced`, { method: 'POST', body: formData }).then(r => r.json())
                ]);

                displayResults({
                    classification: clsRes.result,
                    detection: detRes.result,
                    segmentation: segRes.result,
                    advanced: advRes.result
                });
            }

            showToast('تم التحليل بنجاح', 'success');
        } catch (e) {
            console.error(e);
            showToast('حدث خطأ في التحليل', 'error');
        } finally {
            elements.analyzeBtn.disabled = false;
            elements.analyzeBtn.innerHTML = '🔍 بدء التحليل';
        }
    });

    document.getElementById('clearImage').addEventListener('click', () => {
        state.currentFile = null;
        elements.previewArea.style.display = 'none';
        elements.uploadArea.style.display = 'block';
        elements.resultsContainer.style.display = 'none';
        // Clean up video preview if exists
        const oldVideo = elements.previewArea.querySelector('video');
        if (oldVideo) oldVideo.remove();
        elements.previewImage.style.display = 'block';
    });
}

function handleFile(file) {
    if (!file) return;
    state.currentFile = file;

    // Clear old video if exists
    const oldVideo = elements.previewArea.querySelector('video');
    if (oldVideo) oldVideo.remove();

    if (file.type.startsWith('video/')) {
        elements.previewImage.style.display = 'none';
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.controls = true;
        video.style.width = '100%';
        video.style.borderRadius = '12px';
        elements.previewImage.parentNode.appendChild(video);
    } else {
        elements.previewImage.style.display = 'block';
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    elements.uploadArea.style.display = 'none';
    elements.previewArea.style.display = 'block';
}

function displayVideoResults(result) {
    elements.resultsContainer.style.display = 'block';
    elements.resultsContainer.scrollIntoView({ behavior: 'smooth' });

    // Hide image-specific cards
    elements.detectionResult.style.display = 'none';
    elements.segmentationResult.style.display = 'none';
    elements.advancedAnalysisResult.style.display = 'none';

    // Update Classification Card with Video Summary
    elements.classificationResult.style.display = 'block';
    elements.diseaseName.textContent = result.is_fake ? 'FAKE' : 'REAL';
    elements.diseaseNameAr.textContent = result.status_ar;

    const statusText = result.is_fake ? '⚠️ فيديو مزيف' : '✅ فيديو حقيقي';
    const statusClass = result.is_fake ? 'status-badge diseased' : 'status-badge healthy';
    elements.healthStatus.innerHTML = `<span class="${statusClass}">${statusText}</span>`;

    const conf = Math.round(result.confidence * 100);
    elements.confidenceValue.textContent = `${conf}%`;
    elements.confidenceFill.style.width = `${conf}%`;

    showToast(`تم تحليل ${result.frames_analyzed} إطارات من الفيديو`, 'info');
}

function displayResults(data) {
    elements.resultsContainer.style.display = 'block';
    elements.resultsContainer.scrollIntoView({ behavior: 'smooth' });

    // Classification
    if (data.classification) {
        const cls = data.classification;
        elements.classificationResult.style.display = 'block';
        elements.diseaseName.textContent = cls.class_name;
        elements.diseaseNameAr.textContent = cls.class_name_ar;

        const isFake = cls.class_name === 'FAKE';
        const statusText = isFake ? '⚠️ صورة مزيفة' : '✅ صورة حقيقية';
        const statusClass = isFake ? 'status-badge diseased' : 'status-badge healthy';

        elements.healthStatus.innerHTML = `<span class="${statusClass}">${statusText}</span>`;

        const conf = Math.round(cls.confidence * 100);
        elements.confidenceValue.textContent = `${conf}%`;
        elements.confidenceFill.style.width = `${conf}%`;
    }

    // Detection
    if (data.detection) {
        elements.detectionResult.style.display = 'block';
        elements.numDetections.textContent = data.detection.num_detections;
        elements.detectionImage.src = data.detection.annotated_image;
    }

    // Segmentation (ELA)
    if (data.segmentation) {
        elements.segmentationResult.style.display = 'block';
        elements.diseasePercentage.textContent = `${data.segmentation.fake_percentage}%`;
        elements.maskImage.src = data.segmentation.mask_image;
        elements.overlayImage.src = data.segmentation.overlay_image;
    }

    // Advanced
    if (data.advanced) {
        elements.advancedAnalysisResult.style.display = 'block';
        const adv = data.advanced;
        elements.noiseVar.textContent = adv.noise_analysis.noise_variance;
        elements.blurScore.textContent = adv.quality_metrics.blur_score;
        elements.fftImage.src = adv.frequency_analysis.fft_image;
        elements.elaImage.src = adv.ela_analysis.ela_image;
    }
}

// Camera Logic
function initCamera() {
    elements.startCamera.addEventListener('click', async () => {
        try {
            state.stream = await navigator.mediaDevices.getUserMedia({ video: true });
            elements.cameraVideo.srcObject = state.stream;
            elements.startCamera.style.display = 'none';
            elements.stopCamera.style.display = 'inline-block';
            elements.liveResults.style.display = 'block';

            state.websocket = new WebSocket(WS_URL);
            state.websocket.onopen = () => processFrame();
            state.websocket.onmessage = (msg) => {
                const data = JSON.parse(msg.data);
                if (data.classification) {
                    elements.liveClassName.textContent = data.status_ar;
                    elements.liveConfidence.textContent = `${Math.round(data.classification.confidence * 100)}%`;
                    elements.liveClassName.style.color = data.classification.class_name === 'FAKE' ? 'red' : 'green';
                }
                if (data.fps) elements.fpsValue.textContent = data.fps;

                // Draw boxes
                if (data.detections) {
                    const ctx = elements.cameraCanvas.getContext('2d');
                    elements.cameraCanvas.width = elements.cameraVideo.videoWidth;
                    elements.cameraCanvas.height = elements.cameraVideo.videoHeight;
                    elements.cameraCanvas.style.display = 'block';
                    ctx.clearRect(0, 0, elements.cameraCanvas.width, elements.cameraCanvas.height);

                    data.detections.forEach(det => {
                        const { x, y, width, height } = det.box;
                        ctx.strokeStyle = '#00f2ff';
                        ctx.lineWidth = 3;
                        ctx.strokeRect(x, y, width, height);
                    });
                }

                if (state.stream) requestAnimationFrame(processFrame);
            };
        } catch (e) {
            console.error(e);
            showToast('فشل تشغيل الكاميرا', 'error');
        }
    });

    elements.stopCamera.addEventListener('click', () => {
        if (state.stream) state.stream.getTracks().forEach(t => t.stop());
        if (state.websocket) state.websocket.close();
        state.stream = null;
        elements.cameraVideo.srcObject = null;
        elements.startCamera.style.display = 'inline-block';
        elements.stopCamera.style.display = 'none';
        elements.liveResults.style.display = 'none';
        elements.cameraCanvas.style.display = 'none';
    });
}

function processFrame() {
    if (!state.stream || !state.websocket || state.websocket.readyState !== WebSocket.OPEN) return;

    const canvas = document.createElement('canvas');
    canvas.width = elements.cameraVideo.videoWidth;
    canvas.height = elements.cameraVideo.videoHeight;
    canvas.getContext('2d').drawImage(elements.cameraVideo, 0, 0);

    canvas.toBlob(blob => state.websocket.send(blob), 'image/jpeg', 0.5);
}

// Comparison Logic
function initComparison() {
    elements.compareBox1.addEventListener('click', () => elements.compareInput1.click());
    elements.compareBox2.addEventListener('click', () => elements.compareInput2.click());

    elements.compareInput1.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            elements.compareImg1.src = URL.createObjectURL(file);
            elements.compareImg1.style.display = 'block';
        }
    });

    elements.compareInput2.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            elements.compareImg2.src = URL.createObjectURL(file);
            elements.compareImg2.style.display = 'block';
        }
    });

    elements.startComparison.addEventListener('click', async () => {
        const f1 = elements.compareInput1.files[0];
        const f2 = elements.compareInput2.files[0];
        if (!f1 || !f2) return showToast('اختر الصورتين أولاً', 'error');

        const fd = new FormData();
        fd.append('file1', f1);
        fd.append('file2', f2);

        const res = await fetch(`${API_BASE_URL}/api/compare`, { method: 'POST', body: fd }).then(r => r.json());
        if (res.success) {
            elements.comparisonResults.style.display = 'block';
            elements.compHeadline.textContent = res.result.summary.summary_ar;
            elements.diffImg.src = res.result.diff_image;
        }
    });
}

// Chat Logic
function initChat() {
    elements.toggleChat.addEventListener('click', () => {
        elements.chatbotWidget.style.display = elements.chatbotWidget.style.display === 'flex' ? 'none' : 'flex';
    });

    elements.closeChat.addEventListener('click', () => {
        elements.chatbotWidget.style.display = 'none';
    });

    elements.sendChat.addEventListener('click', async () => {
        const msg = elements.chatInput.value;
        if (!msg) return;

        addMsg(msg, 'user');
        elements.chatInput.value = '';

        const res = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        }).then(r => r.json());

        addMsg(res.response, 'bot');
    });
}

function addMsg(text, sender) {
    const div = document.createElement('div');
    div.className = `msg ${sender}`;
    div.textContent = text;
    elements.chatMessages.appendChild(div);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function showToast(msg, type = 'info') {
    elements.toast.textContent = msg;
    elements.toast.className = `toast ${type} show`;
    setTimeout(() => elements.toast.className = 'toast', 3000);
}
