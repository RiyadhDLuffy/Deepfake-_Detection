# تقرير مشروع TruthLens: نظام متطور لكشف التزييف العميق (Deepfake Detection)

## 1. ملخص المشروع (Project Summary)
نظام **TruthLens** هو منصة أمنية وتربوية تهدف إلى مواجهة تحديات التزييف الرقمي. يستخدم النظام خوارزميات التعلم العميق (Deep Learning) والتحليل الجنائي للصور (Digital Image Forensics) لتحديد ما إذا كانت الصور أو الفيديوهات حقيقية أم مولدة بواسطة الذكاء الاصطناعي.

---

## 2. المميزات الرئيسية (Key Features)
*   **تصنيف دقيق (Deep Classification):** استخدام شبكات MobileNetV2 المدربة للكشف عن آثار التوليد الاصطناعي.
*   **كشف الوجوه (Face Detection):** دمج YOLOv8 لتحديد الوجوه ومناطق الاهتمام بدقة عالية وسرعة فائقة.
*   **التحليل الجنائي (Forensic ELA):** تطبيق "تحليل مستوى الخطأ" (Error Level Analysis) لكشف التلاعب في ضغط الصورة (Re-compression artifacts).
*   **تحليل الفيديو (Video Analysis):** معالجة ملفات الفيديو وفحص إطاراتها بشكل آلي.
*   **المعالجة المباشرة (Real-time Stream):** دعم البث المباشر عبر الكاميرا (Webcam) باستخدام WebSockets.
*   **واجهة مستخدم احترافية (Premium UI):** تصميم عصري (Dark Mode) يدعم اللغة العربية بالكامل.

---

## 3. الهندسة التقنية (Technical Architecture)

### الفرونت-أند (Frontend)
*   **التقنيات:** Vanilla JS, HTML5, CSS3.
*   **التصميم:** Responsive Design, Glassmorphism, Micro-animations.
*   **المميزات:** تدعم العمل كتطبيق ويب تقدمي (PWA).

### الباك-أند (Backend)
*   **Framework:** FastAPI (Python).
*   **AI Engines:**
    *   **PyTorch:** لتشغيل موديلات التصنيف.
    *   **Ultralytics (YOLOv8):** لكشف الوجوه والأشخاص.
    *   **Torchvision:** لموديلات التقطيع (Segmentation - DeepLabV3).
    *   **OpenCV:** لمعالجة الصور والتحليل الجنائي.

---

## 4. الموديلات المستخدمة (AI Models)
1.  **MobileNetV2:** تم تخصيصها للعمل كمصنف (Binary Classifier: Real vs Fake) بفضل خفتها وسرعتها في التنفيذ.
2.  **YOLOv8n:** أحدث إصدارات YOLO للكشف اللحظي عن الأجسام.
3.  **DeepLabV3:** لتقطيع الأجزاء المختلفة من الصورة وفحص "البصمة الرقمية" لكل جزء بشكل منفصل.

---

## 5. كيفية التشغيل (Deployment & Installation)
1. تثبيت المتطلبات: `pip install -r backend/requirements.txt`
2. تشغيل الخادم: `python backend/app.py`
3. الدخول عبر المتصفح: `http://localhost:8000`

---

## 6. النتائج والتوصيات (Results & Recommendations)
أظهر النظام دقة عالية في كشف الصور المولدة عبر برامج مثل (Midjourney, Stable Diffusion, DALL-E). نوصي في الإصدارات القادمة بزيادة قاعدة البيانات لتشمل الفيديوهات المولدة عبر (Sora) والتقنيات الأحدث.

---

**إعداد:** RiyadhDLuffy
**تاريخ:** 2026-01-28
