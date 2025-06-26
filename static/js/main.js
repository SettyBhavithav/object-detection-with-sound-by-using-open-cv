let isStreaming = false;
let streamInterval = null;
let lastAlertTime = 0;
const alertCooldown = 2000; // ms

document.getElementById('conf-slider').addEventListener('input', (e) => {
    document.getElementById('conf-val').innerText = parseFloat(e.target.value).toFixed(2);
});

function switchMode(mode) {
    if (isStreaming) stopWebcam();

    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.view-mode').forEach(view => view.classList.add('hidden'));

    if (mode === 'webcam') {
        document.getElementById('tab-webcam').classList.add('active');
        document.getElementById('webcam-container').classList.remove('hidden');
    } else {
        document.getElementById('tab-upload').classList.add('active');
        document.getElementById('upload-container').classList.remove('hidden');
    }
}

async function startWebcam() {
    const video = document.getElementById('webcam-feed');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        isStreaming = true;
        
        document.getElementById('start-cam-btn').disabled = true;
        document.getElementById('stop-cam-btn').disabled = false;
        document.getElementById('status-text').innerText = "Streaming Web Camera";

        streamInterval = setInterval(captureAndDetect, 150); // ~7 FPS stream rate to server
    } catch (err) {
        alert("Webcam Access Failed: " + err.message);
    }
}

function stopWebcam() {
    const video = document.getElementById('webcam-feed');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    isStreaming = false;
    clearInterval(streamInterval);

    document.getElementById('start-cam-btn').disabled = false;
    document.getElementById('stop-cam-btn').disabled = true;
    document.getElementById('status-text').innerText = "System Ready";

    const canvas = document.getElementById('detection-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

let isProcessingFrame = false;

async function captureAndDetect() {
    if (!isStreaming || isProcessingFrame) return;

    const video = document.getElementById('webcam-feed');
    if (!video || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return;

    const canvas = document.getElementById('detection-canvas');
    const ctx = canvas.getContext('2d');

    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;

    isProcessingFrame = true;

    // Capture current frame as blob
    const offscreen = document.createElement('canvas');
    offscreen.width = canvas.width;
    offscreen.height = canvas.height;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

    offscreen.toBlob(async (blob) => {
        if (!blob) {
            isProcessingFrame = false;
            return;
        }

        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');
        formData.append('conf', document.getElementById('conf-slider').value);
        formData.append('roi', document.getElementById('roi-toggle').checked);

        const startTime = performance.now();
        try {
            const response = await fetch('/api/detect', { method: 'POST', body: formData });
            const data = await response.json();
            const elapsed = performance.now() - startTime;

            if (data.status === 'success' && Array.isArray(data.detections)) {
                updateStats(data.detections, (1000 / elapsed).toFixed(1));
                renderCanvasOverlays(ctx, data.detections, canvas.width, canvas.height);
                checkAudioAlerts(data.detections);
            }
        } catch (err) {
            console.error("Frame processing error:", err);
        } finally {
            isProcessingFrame = false;
        }
    }, 'image/jpeg', 0.7);
}

function renderCanvasOverlays(ctx, detections, width, height) {
    ctx.clearRect(0, 0, width, height);

    // Draw ROI if enabled
    if (document.getElementById('roi-toggle').checked) {
        const rx = width * 0.25, ry = height * 0.2, rw = width * 0.5, rh = height * 0.6;
        ctx.strokeStyle = '#da3633';
        ctx.lineWidth = 3;
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.fillStyle = '#da3633';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.fillText('DANGER ZONE (ROI)', rx + 5, ry + 20);
    }

    detections.forEach(det => {
        const [x, y, w, h] = det.box;

        ctx.save();
        // Glowing stroke for high visibility
        ctx.strokeStyle = '#00ff66';
        ctx.lineWidth = 4;
        ctx.shadowColor = '#00ff66';
        ctx.shadowBlur = 12;
        ctx.strokeRect(x, y, w, h);

        // Semi-transparent box overlay
        ctx.fillStyle = 'rgba(0, 255, 102, 0.15)';
        ctx.fillRect(x, y, w, h);

        // Label Badge
        const text = ` ${det.class_name.toUpperCase()} ${(det.confidence * 100).toFixed(0)}% `;
        ctx.font = 'bold 15px Inter, sans-serif';
        const textWidth = ctx.measureText(text).width;

        ctx.shadowBlur = 0;
        ctx.fillStyle = '#00ff66';
        ctx.fillRect(x, Math.max(0, y - 28), textWidth + 10, 28);

        ctx.fillStyle = '#000000';
        ctx.fillText(text, x + 5, Math.max(19, y - 9));
        ctx.restore();
    });
}

function checkAudioAlerts(detections) {
    if (detections.length === 0) return;

    const soundEnabled = document.getElementById('audio-toggle').checked;
    const ttsEnabled = document.getElementById('tts-toggle').checked;

    if (!soundEnabled && !ttsEnabled) return;

    const now = Date.now();
    if (now - lastAlertTime < alertCooldown) return;

    lastAlertTime = now;

    if (soundEnabled) {
        document.getElementById('chime-audio').play().catch(e => console.log(e));
    }

    if (ttsEnabled && 'speechSynthesis' in window) {
        const targetClass = detections[0].class_name;
        const msg = new SpeechSynthesisUtterance(`${targetClass} detected`);
        msg.rate = 1.1;
        window.speechSynthesis.speak(msg);
    }
}

function updateStats(detections, fps) {
    document.getElementById('stat-count').innerText = detections.length;
    document.getElementById('stat-fps').innerText = fps;

    const list = document.getElementById('detection-list');
    list.innerHTML = '';

    if (detections.length === 0) {
        list.innerHTML = '<li class="empty-msg">No active detections</li>';
        return;
    }

    detections.forEach(det => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${det.class_name.toUpperCase()}</span><strong>${(det.confidence * 100).toFixed(0)}%</strong>`;
        list.appendChild(li);
    });
}

async function handleFileUpload(files) {
    if (files.length === 0) return;

    const file = files[0];
    const formData = new FormData();
    formData.append('image', file);
    formData.append('conf', document.getElementById('conf-slider').value);
    formData.append('roi', document.getElementById('roi-toggle').checked);

    document.getElementById('status-text').innerText = "Processing File...";

    try {
        const response = await fetch('/api/detect_render_image', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.image_base64) {
            document.getElementById('upload-result').classList.remove('hidden');
            document.getElementById('result-img').src = 'data:image/jpeg;base64,' + data.image_base64;
            updateStats(data.detections, '--');
            document.getElementById('status-text').innerText = "Detection Complete";
        }
    } catch (err) {
        alert("Upload error: " + err.message);
        document.getElementById('status-text').innerText = "System Ready";
    }
}
