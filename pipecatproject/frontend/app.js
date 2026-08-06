console.log("frontend app.js loaded");
document.addEventListener("DOMContentLoaded", () => {
  const startButton = document.getElementById("startButton");
  const stopCameraButton = document.getElementById("stopCameraButton");
  const stopMicrophoneButton = document.getElementById("stopMicrophoneButton");
  const stopSessionButton = document.getElementById("stopSessionButton");
  const captureImageButton = document.getElementById("captureImageButton");
  const uploadImageButton = document.getElementById("uploadImageButton");
  const sendFrameButton = document.getElementById("sendFrameButton");
  const sendTextButton = document.getElementById("sendTextButton");
  const imageUploadInput = document.getElementById("imageUploadInput");
  const useCapturedImageRadio = document.getElementById("useCapturedImageRadio");
  const useUploadedImageRadio = document.getElementById("useUploadedImageRadio");
  const statusMessage = document.getElementById("statusMessage");
  const responseText = document.getElementById("responseText");
  const emotionMetadata = document.getElementById("emotionMetadata");
  const messageInput = document.getElementById("messageInput");
  const voiceTranscript = document.getElementById("voiceTranscript");
  const camera = document.getElementById("camera");
  const imagePreview = document.getElementById("imagePreview");
  const imageSourceLabel = document.getElementById("imageSourceLabel");

  const startRecordingButton = document.getElementById("startRecordingButton");
  const stopRecordingButton = document.getElementById("stopRecordingButton");
  const sendVoiceButton = document.getElementById("sendVoiceButton");
  const sendMultimodalButton = document.getElementById("sendMultimodalButton");
  const startCameraButton = document.getElementById("startCameraButton");
  const startMicrophoneButton = document.getElementById("startMicrophoneButton");
  const clearCapturedFrameButton = document.getElementById("clearCapturedFrameButton");

  // ── Streaming toggle buttons (added dynamically below) ──────────────────
  const streamVideoBtn = document.getElementById("streamVideoButton");
  const streamAudioBtn = document.getElementById("streamAudioButton");

  let localStream = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let capturedImageBytes = null;
  let uploadedImageBytes = null;
  let capturedImageUrl = null;
  let uploadedImageUrl = null;
  let activeImageSource = null; // 'captured' | 'uploaded' | null

  // ── WebSocket state ──────────────────────────────────────────────────────
  let videoWs = null;
  let audioWs = null;
  let videoStreamInterval = null;   // setInterval handle for frame capture
  let audioStreamRecorder = null;   // MediaRecorder for streaming audio
  let isStreamingVideo = false;
  let isStreamingAudio = false;

  // Build WebSocket URL from current page origin (works on localhost and Render)
  function wsUrl(path) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}${path}`;
  }

  function setStatus(message) {
    statusMessage.textContent = message;
  }

  function updateButtonStates() {
    const hasStream = Boolean(localStream);
    const hasVideo = hasStream && localStream.getVideoTracks().some((track) => track.readyState !== "ended");
    const hasAudio = hasStream && localStream.getAudioTracks().some((track) => track.readyState !== "ended");
    const isRecording = mediaRecorder?.state === "recording";

    stopCameraButton.disabled = !hasVideo;
    stopMicrophoneButton.disabled = !hasAudio;
    stopSessionButton.disabled = !hasStream;
    captureImageButton.disabled = !hasVideo;
    uploadImageButton.disabled = !imageUploadInput.files.length;
    startRecordingButton.disabled = !hasAudio || isRecording;
    stopRecordingButton.disabled = !isRecording;
    sendVoiceButton.disabled = isRecording || audioChunks.length === 0;
    sendFrameButton.disabled = !capturedImageBytes;
    sendMultimodalButton.disabled = isRecording || audioChunks.length === 0 || !messageInput.value.trim() || !activeImageSource;
    startCameraButton.disabled = hasVideo;
    startMicrophoneButton.disabled = hasAudio;

    if (streamVideoBtn) {
      streamVideoBtn.disabled = !hasVideo;
      streamVideoBtn.textContent = isStreamingVideo ? "Stop Video Stream" : "Stream Video";
      streamVideoBtn.classList.toggle("rail-btn-danger", isStreamingVideo);
    }
    if (streamAudioBtn) {
      streamAudioBtn.disabled = !hasAudio;
      streamAudioBtn.textContent = isStreamingAudio ? "Stop Audio Stream" : "Stream Audio";
      streamAudioBtn.classList.toggle("rail-btn-danger", isStreamingAudio);
    }
  }

  // ── Video WebSocket streaming ────────────────────────────────────────────
  function startVideoStream() {
    if (isStreamingVideo) { stopVideoStream(); return; }
    if (!localStream || !localStream.getVideoTracks().some(t => t.readyState !== "ended")) {
      setStatus("Start the camera before streaming video.");
      return;
    }

    videoWs = new WebSocket(wsUrl("/ws/video"));
    videoWs.binaryType = "arraybuffer";

    videoWs.onopen = () => {
      isStreamingVideo = true;
      setStatus("Video stream started.");
      updateButtonStates();

      // Send a JPEG frame every 500 ms
      videoStreamInterval = setInterval(async () => {
        if (videoWs?.readyState !== WebSocket.OPEN) return;
        const blob = await grabFrameBlob();
        if (!blob) return;
        const buf = await blob.arrayBuffer();
        videoWs.send(buf);
      }, 500);
    };

    videoWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.face_emotion) {
          // Merge into displayed metadata
          let current = {};
          try { current = JSON.parse(emotionMetadata.textContent); } catch (_) {}
          current.face_emotion = data.face_emotion;
          emotionMetadata.textContent = JSON.stringify(current, null, 2);
        }
      } catch (_) {}
    };

    videoWs.onerror = () => setStatus("Video WebSocket error.");
    videoWs.onclose = () => {
      isStreamingVideo = false;
      clearInterval(videoStreamInterval);
      videoStreamInterval = null;
      updateButtonStates();
    };
  }

  function stopVideoStream() {
    clearInterval(videoStreamInterval);
    videoStreamInterval = null;
    if (videoWs) { videoWs.close(); videoWs = null; }
    isStreamingVideo = false;
    setStatus("Video stream stopped.");
    updateButtonStates();
  }

  // ── Audio WebSocket streaming ────────────────────────────────────────────
  function startAudioStream() {
    if (isStreamingAudio) { stopAudioStream(); return; }
    const audioTracks = localStream?.getAudioTracks().filter(t => t.readyState !== "ended") || [];
    if (!audioTracks.length) {
      setStatus("Start the microphone before streaming audio.");
      return;
    }

    audioWs = new WebSocket(wsUrl("/ws/audio"));
    audioWs.binaryType = "arraybuffer";

    audioWs.onopen = () => {
      isStreamingAudio = true;
      setStatus("Audio stream started.");
      updateButtonStates();

      // Record in 3-second chunks and send each chunk
      const audioStream = new MediaStream([audioTracks[0]]);
      let opts = {};
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) opts.mimeType = "audio/webm;codecs=opus";
      else if (MediaRecorder.isTypeSupported("audio/webm")) opts.mimeType = "audio/webm";

      audioStreamRecorder = new MediaRecorder(audioStream, opts);
      audioStreamRecorder.ondataavailable = async (e) => {
        if (!e.data || e.data.size === 0 || audioWs?.readyState !== WebSocket.OPEN) return;
        const buf = await e.data.arrayBuffer();
        audioWs.send(buf);
      };
      audioStreamRecorder.start(3000); // timeslice: fire ondataavailable every 3 s
    };

    audioWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.transcript) voiceTranscript.textContent = data.transcript;
        if (data.response) responseText.textContent = data.response;
        if (data.metadata) {
          const meta = data.metadata;
          emotionMetadata.textContent = JSON.stringify({
            source: meta.source,
            text_emotion: meta.text_emotion,
            fused_emotion: meta.fused_emotion,
          }, null, 2);
        }
        setStatus("Audio stream: response received.");
      } catch (_) {}
    };

    audioWs.onerror = () => setStatus("Audio WebSocket error.");
    audioWs.onclose = () => {
      isStreamingAudio = false;
      if (audioStreamRecorder?.state === "recording") audioStreamRecorder.stop();
      audioStreamRecorder = null;
      updateButtonStates();
    };
  }

  function stopAudioStream() {
    if (audioStreamRecorder?.state === "recording") audioStreamRecorder.stop();
    audioStreamRecorder = null;
    if (audioWs) { audioWs.close(); audioWs = null; }
    isStreamingAudio = false;
    setStatus("Audio stream stopped.");
    updateButtonStates();
  }

  // ── Image source helpers ─────────────────────────────────────────────────
  function getExplicitImageSourceSelection() {
    if (useCapturedImageRadio.checked) return "captured";
    if (useUploadedImageRadio.checked) return "uploaded";
    return null;
  }

  function refreshPreviewForSelectedImageSource() {
    if (activeImageSource === "captured" && capturedImageUrl) {
      imagePreview.src = capturedImageUrl;
      imageSourceLabel.textContent = "Captured frame";
      imagePreview.style.display = "block";
      clearCapturedFrameButton.style.display = "inline-flex";
    } else if (activeImageSource === "uploaded" && uploadedImageUrl) {
      imagePreview.src = uploadedImageUrl;
      imageSourceLabel.textContent = "Uploaded image";
      imagePreview.style.display = "block";
      clearCapturedFrameButton.style.display = "none";
    } else {
      imagePreview.src = "";
      imageSourceLabel.textContent = "No image selected";
      imagePreview.style.display = "none";
      clearCapturedFrameButton.style.display = "none";
    }
  }

  function syncImageSourceControls() {
    const hasCaptured = Boolean(capturedImageBytes);
    const hasUploaded = Boolean(uploadedImageBytes);
    useCapturedImageRadio.disabled = !hasCaptured;
    useUploadedImageRadio.disabled = !hasUploaded;

    const explicitSelection = getExplicitImageSourceSelection();
    if (hasCaptured && !hasUploaded) {
      activeImageSource = "captured";
      useCapturedImageRadio.checked = true;
    } else if (!hasCaptured && hasUploaded) {
      activeImageSource = "uploaded";
      useUploadedImageRadio.checked = true;
    } else if (hasCaptured && hasUploaded) {
      if (explicitSelection) {
        activeImageSource = explicitSelection;
      } else if (activeImageSource !== "captured" && activeImageSource !== "uploaded") {
        activeImageSource = "captured";
        useCapturedImageRadio.checked = true;
      }
    } else {
      activeImageSource = null;
      useCapturedImageRadio.checked = false;
      useUploadedImageRadio.checked = false;
    }

    refreshPreviewForSelectedImageSource();
  }

  function clearCapturedFrame() {
    if (capturedImageUrl) URL.revokeObjectURL(capturedImageUrl);
    capturedImageBytes = null;
    capturedImageUrl = null;

    if (uploadedImageBytes) {
      activeImageSource = "uploaded";
      useUploadedImageRadio.checked = true;
      useCapturedImageRadio.checked = false;
    } else {
      activeImageSource = null;
      useCapturedImageRadio.checked = false;
      useUploadedImageRadio.checked = false;
    }

    refreshPreviewForSelectedImageSource();
    updateButtonStates();
    setStatus("Captured frame canceled.");
  }

  function handleImageSourceSelectionChange() {
    if (useCapturedImageRadio.checked && capturedImageBytes) activeImageSource = "captured";
    else if (useUploadedImageRadio.checked && uploadedImageBytes) activeImageSource = "uploaded";
    refreshPreviewForSelectedImageSource();
    updateButtonStates();
  }

  // ── Camera / microphone controls ─────────────────────────────────────────
  async function startCamera() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      if (!devices.some(d => d.kind === "videoinput")) {
        setStatus("No camera device found."); updateButtonStates(); return;
      }
      const videoStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "user" },
      });
      if (localStream) videoStream.getVideoTracks().forEach(t => localStream.addTrack(t));
      else localStream = videoStream;
      camera.srcObject = localStream;
      camera.style.display = "block";
      await camera.play().catch(() => {});
      setStatus("Camera started.");
    } catch (err) {
      if (err.name === "NotAllowedError") setStatus("Camera access denied.");
      else if (err.name === "NotFoundError") setStatus("No camera found.");
      else setStatus(`Could not start camera: ${err.message}`);
    }
    updateButtonStates();
  }

  async function startMicrophone() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      if (!devices.some(d => d.kind === "audioinput")) {
        setStatus("No microphone found."); updateButtonStates(); return;
      }
      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (localStream) audioStream.getAudioTracks().forEach(t => localStream.addTrack(t));
      else localStream = audioStream;
      setStatus("Microphone started.");
    } catch (err) {
      if (err.name === "NotAllowedError") setStatus("Microphone access denied.");
      else if (err.name === "NotFoundError") setStatus("No microphone found.");
      else setStatus(`Could not start microphone: ${err.message}`);
    }
    updateButtonStates();
  }

  async function startSession() {
    try {
      localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
      camera.srcObject = localStream;
      setStatus("Camera and microphone connected.");
    } catch (error) {
      setStatus(`Could not start session: ${error.message}`);
      localStream = null;
    }
    updateButtonStates();
  }

  function stopCamera() {
    if (!localStream) { setStatus("No active session."); return; }
    const tracks = localStream.getVideoTracks().filter(t => t.readyState !== "ended");
    if (!tracks.length) { setStatus("Camera already stopped."); return; }
    if (isStreamingVideo) stopVideoStream();
    tracks.forEach(t => { t.stop(); localStream.removeTrack(t); });
    camera.srcObject = null;
    setStatus("Camera stopped.");
    if (!localStream.getTracks().length) localStream = null;
    updateButtonStates();
  }

  function stopMicrophone() {
    if (!localStream) { setStatus("No active session."); return; }
    const tracks = localStream.getAudioTracks().filter(t => t.readyState !== "ended");
    if (!tracks.length) { setStatus("Microphone already stopped."); return; }
    if (isStreamingAudio) stopAudioStream();
    tracks.forEach(t => { t.stop(); localStream.removeTrack(t); });
    setStatus("Microphone stopped.");
    if (!localStream.getTracks().length) { localStream = null; camera.srcObject = null; }
    updateButtonStates();
  }

  function stopSession() {
    if (!localStream) { setStatus("No active session."); return; }
    if (isStreamingVideo) stopVideoStream();
    if (isStreamingAudio) stopAudioStream();
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
    camera.srcObject = null;
    setStatus("Session stopped.");
    updateButtonStates();
  }

  // ── Frame capture helpers ────────────────────────────────────────────────
  async function grabFrameBlob() {
    if (!localStream || !localStream.getVideoTracks().some(t => t.readyState !== "ended")) return null;
    const track = localStream.getVideoTracks()[0];
    let blob;
    if (typeof ImageCapture === "function") {
      try {
        const ic = new ImageCapture(track);
        const photo = await ic.takePhoto();
        blob = new Blob([await photo.arrayBuffer()], { type: "image/jpeg" });
      } catch (_) {}
    }
    if (!blob) {
      if (!camera.videoWidth || !camera.videoHeight) return null;
      const canvas = document.createElement("canvas");
      canvas.width = camera.videoWidth; canvas.height = camera.videoHeight;
      canvas.getContext("2d").drawImage(camera, 0, 0);
      blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.95));
    }
    return blob || null;
  }

  function setCapturedImage(blob, arrayBuffer) {
    capturedImageBytes = arrayBuffer;
    if (capturedImageUrl) URL.revokeObjectURL(capturedImageUrl);
    capturedImageUrl = URL.createObjectURL(blob);
    activeImageSource = "captured";
    useCapturedImageRadio.checked = true;
    syncImageSourceControls();
  }

  async function captureFace() {
    if (!localStream || !localStream.getVideoTracks().length) {
      setStatus("Start the session before capturing a frame."); return;
    }
    const blob = await grabFrameBlob();
    if (!blob) { setStatus("Camera feed is not ready yet."); return; }
    setCapturedImage(blob, await blob.arrayBuffer());
    setStatus("Frame extracted.");
    updateButtonStates();
  }

  // ── HTTP API calls ───────────────────────────────────────────────────────
  async function analyzeUploadedImage() {
    const file = imageUploadInput.files[0];
    if (!file) { setStatus("Select an image before analyzing."); return; }
    try {
      const arrayBuffer = uploadedImageBytes || (await file.arrayBuffer());
      uploadedImageBytes = arrayBuffer;
      if (!uploadedImageUrl) uploadedImageUrl = URL.createObjectURL(file);
      syncImageSourceControls();
      const response = await fetch("/api/image", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: arrayBuffer,
      });
      if (!response.ok) { setStatus(`Image analysis failed: ${await response.text()}`); return; }
      const data = await response.json();
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify({ source: meta.source, text_emotion: meta.text_emotion, fused_emotion: meta.fused_emotion }, null, 2);
      setStatus("Uploaded image analyzed.");
    } catch (error) {
      setStatus(`Image analysis failed: ${error.message}`);
    }
    updateButtonStates();
  }

  async function updateUploadedImagePreview() {
    const file = imageUploadInput.files[0];
    if (!file) {
      uploadedImageBytes = null;
      if (uploadedImageUrl) { URL.revokeObjectURL(uploadedImageUrl); uploadedImageUrl = null; }
      syncImageSourceControls(); updateButtonStates(); return;
    }
    if (uploadedImageUrl) URL.revokeObjectURL(uploadedImageUrl);
    uploadedImageUrl = URL.createObjectURL(file);
    uploadedImageBytes = await file.arrayBuffer();
    if (!capturedImageBytes && !activeImageSource) { activeImageSource = "uploaded"; useUploadedImageRadio.checked = true; }
    if (activeImageSource === "uploaded") {
      imagePreview.src = uploadedImageUrl;
      imageSourceLabel.textContent = "Uploaded image";
      imagePreview.style.display = "block";
    }
    syncImageSourceControls();
    setStatus("Image selected.");
    updateButtonStates();
  }

  async function sendFrame() {
    if (!capturedImageBytes) { setStatus("Extract a frame first."); return; }
    try {
      const response = await fetch("/api/image", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: capturedImageBytes,
      });
      if (!response.ok) { setStatus(`Frame request failed: ${await response.text()}`); return; }
      const data = await response.json();
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify({ source: meta.source, text_emotion: meta.text_emotion, fused_emotion: meta.fused_emotion }, null, 2);
      setStatus("Captured frame sent and analyzed.");
    } catch (error) {
      setStatus(`Frame request failed: ${error.message}`);
    }
    updateButtonStates();
  }

  function createMediaRecorder() {
    const tracks = localStream?.getAudioTracks().filter(t => t.readyState !== "ended") || [];
    if (!tracks.length) return null;
    const audioStream = new MediaStream([tracks[0]]);
    let opts = {};
    if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) opts.mimeType = "audio/webm;codecs=opus";
    else if (MediaRecorder.isTypeSupported("audio/webm")) opts.mimeType = "audio/webm";
    try {
      const recorder = new MediaRecorder(audioStream, opts);
      recorder.ondataavailable = (e) => {
        if (e.data?.size > 0) { audioChunks.push(e.data); updateButtonStates(); }
      };
      recorder.onstop = () => { setStatus("Recording stopped. Ready to send."); updateButtonStates(); };
      recorder.onerror = (e) => setStatus(`Recording error: ${e.error?.message || "Unknown"}`);
      return recorder;
    } catch (error) {
      setStatus(`Could not create recorder: ${error.message}`); return null;
    }
  }

  async function startVoiceRecording() {
    if (!localStream || !localStream.getAudioTracks().length) {
      setStatus("Start the session before recording."); return;
    }
    audioChunks = [];
    mediaRecorder = createMediaRecorder();
    if (!mediaRecorder) return;
    try { mediaRecorder.start(); setStatus("Recording voice..."); }
    catch (err) { setStatus(`Could not start recording: ${err.message}`); mediaRecorder = null; }
    updateButtonStates();
  }

  function stopVoiceRecording() {
    if (!mediaRecorder || mediaRecorder.state !== "recording") {
      setStatus("No active recording."); return;
    }
    try { mediaRecorder.stop(); } catch (err) { setStatus(`Stop failed: ${err.message}`); }
    updateButtonStates();
  }

  async function sendVoiceInput() {
    if (!audioChunks.length) { setStatus("Record audio before sending."); return; }
    try {
      const mimeType = mediaRecorder?.mimeType || "audio/webm";
      const blob = new Blob(audioChunks, { type: mimeType });
      const response = await fetch("/api/audio", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: await blob.arrayBuffer(),
      });
      if (!response.ok) { setStatus(`Voice request failed: ${await response.text()}`); return; }
      const data = await response.json();
      voiceTranscript.textContent = data.transcript || "No transcript returned.";
      responseText.textContent = data.response || "";
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify({ source: meta.source, text_emotion: meta.text_emotion, fused_emotion: meta.fused_emotion }, null, 2);
      setStatus("Voice input processed.");
      audioChunks = [];
    } catch (error) {
      setStatus(`Voice request failed: ${error.message}`);
    }
    updateButtonStates();
  }

  async function sendMultimodalInput() {
    if (!messageInput.value.trim()) { setStatus("Enter a message first."); return; }
    if (!audioChunks.length) { setStatus("Record audio first."); return; }
    let imageBytes = activeImageSource === "uploaded" ? uploadedImageBytes
                   : activeImageSource === "captured" ? capturedImageBytes
                   : uploadedImageBytes || capturedImageBytes;
    if (!imageBytes) { setStatus("Provide an image before sending multimodal input."); return; }

    try {
      const mimeType = mediaRecorder?.mimeType || "audio/webm";
      const blob = new Blob(audioChunks, { type: mimeType });
      const formData = new FormData();
      formData.append("text", messageInput.value.trim());
      formData.append("image", new Blob([imageBytes], { type: "image/jpeg" }), "image.jpg");
      formData.append("audio", blob, "audio.webm");
      const response = await fetch("/api/multimodal", { method: "POST", body: formData });
      if (!response.ok) { setStatus(`Multimodal request failed: ${await response.text()}`); return; }
      const data = await response.json();
      voiceTranscript.textContent = data.transcript || "No transcript returned.";
      responseText.textContent = data.response || "";
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify({
        source: meta.source, face_emotion: meta.face_emotion,
        speech_sentiment: meta.speech_sentiment, text_emotion: meta.text_emotion, fused_emotion: meta.fused_emotion,
      }, null, 2);
      setStatus("Multimodal input processed.");
      audioChunks = [];
    } catch (error) {
      setStatus(`Multimodal request failed: ${error.message}`);
    }
    updateButtonStates();
  }

  async function sendTextMessage() {
    const text = messageInput.value.trim();
    if (!text) { setStatus("Enter a message first."); return; }
    sendTextButton.disabled = true;
    try {
      const response = await fetch("/api/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) { setStatus(`Text request failed: ${await response.text()}`); return; }
      const data = await response.json();
      responseText.textContent = data.response || "";
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify({ source: meta.source, text_emotion: meta.text_emotion, fused_emotion: meta.fused_emotion }, null, 2);
      setStatus("Text sent and response generated.");
    } catch (error) {
      setStatus(`Text request failed: ${error.message}`);
    } finally {
      sendTextButton.disabled = false;
    }
  }

  // ── Event listeners ──────────────────────────────────────────────────────
  startButton.addEventListener("click", startSession);
  startCameraButton.addEventListener("click", startCamera);
  startMicrophoneButton.addEventListener("click", startMicrophone);
  stopCameraButton.addEventListener("click", stopCamera);
  stopMicrophoneButton.addEventListener("click", stopMicrophone);
  stopSessionButton.addEventListener("click", stopSession);
  captureImageButton.addEventListener("click", captureFace);
  uploadImageButton.addEventListener("click", analyzeUploadedImage);
  sendFrameButton.addEventListener("click", sendFrame);
  clearCapturedFrameButton.addEventListener("click", clearCapturedFrame);
  sendTextButton.addEventListener("click", sendTextMessage);
  sendMultimodalButton.addEventListener("click", sendMultimodalInput);
  startRecordingButton.addEventListener("click", startVoiceRecording);
  stopRecordingButton.addEventListener("click", stopVoiceRecording);
  sendVoiceButton.addEventListener("click", sendVoiceInput);
  imageUploadInput.addEventListener("change", updateUploadedImagePreview);
  useCapturedImageRadio.addEventListener("change", handleImageSourceSelectionChange);
  useUploadedImageRadio.addEventListener("change", handleImageSourceSelectionChange);
  messageInput.addEventListener("input", updateButtonStates);

  if (streamVideoBtn) streamVideoBtn.addEventListener("click", startVideoStream);
  if (streamAudioBtn) streamAudioBtn.addEventListener("click", startAudioStream);

  updateButtonStates();
});
  const startButton = document.getElementById("startButton");
  const stopCameraButton = document.getElementById("stopCameraButton");
  const stopMicrophoneButton = document.getElementById("stopMicrophoneButton");
  const stopSessionButton = document.getElementById("stopSessionButton");
  const captureImageButton = document.getElementById("captureImageButton");
  const uploadImageButton = document.getElementById("uploadImageButton");
  const sendFrameButton = document.getElementById("sendFrameButton");
  const sendTextButton = document.getElementById("sendTextButton");
  const imageUploadInput = document.getElementById("imageUploadInput");
  const useCapturedImageRadio = document.getElementById("useCapturedImageRadio");
  const useUploadedImageRadio = document.getElementById("useUploadedImageRadio");
  const statusMessage = document.getElementById("statusMessage");
  const responseText = document.getElementById("responseText");
  const emotionMetadata = document.getElementById("emotionMetadata");
  const messageInput = document.getElementById("messageInput");
  const voiceTranscript = document.getElementById("voiceTranscript");
  const camera = document.getElementById("camera");
  const imagePreview = document.getElementById("imagePreview");
  const imageSourceLabel = document.getElementById("imageSourceLabel");

  const startRecordingButton = document.getElementById("startRecordingButton");
  const stopRecordingButton = document.getElementById("stopRecordingButton");
  const sendVoiceButton = document.getElementById("sendVoiceButton");
  const sendMultimodalButton = document.getElementById("sendMultimodalButton");
  const startCameraButton = document.getElementById("startCameraButton");
  const startMicrophoneButton = document.getElementById("startMicrophoneButton");
  const clearCapturedFrameButton = document.getElementById("clearCapturedFrameButton");
  console.log('startCameraButton element:', startCameraButton, 'startMicrophoneButton element:', startMicrophoneButton);

  let localStream = null;
  let mediaRecorder = null;
  let audioChunks = [];
  let capturedImageBytes = null;
  let uploadedImageBytes = null;
  let capturedImageUrl = null;
  let uploadedImageUrl = null;
  let activeImageSource = null; // 'captured' | 'uploaded' | null

  function setStatus(message) {
    statusMessage.textContent = message;
  }

  function updateButtonStates() {
    const hasStream = Boolean(localStream);
    const hasVideo = hasStream && localStream.getVideoTracks().some((track) => track.readyState !== "ended");
    const hasAudio = hasStream && localStream.getAudioTracks().some((track) => track.readyState !== "ended");
    const isRecording = mediaRecorder?.state === "recording";

    stopCameraButton.disabled = !hasVideo;
    stopMicrophoneButton.disabled = !hasAudio;
    stopSessionButton.disabled = !hasStream;
    captureImageButton.disabled = !hasVideo;
    uploadImageButton.disabled = !imageUploadInput.files.length;
    startRecordingButton.disabled = !hasAudio || isRecording;
    stopRecordingButton.disabled = !isRecording;
    sendVoiceButton.disabled = isRecording || audioChunks.length === 0;
    sendFrameButton.disabled = !capturedImageBytes;
    sendMultimodalButton.disabled = isRecording || audioChunks.length === 0 || !messageInput.value.trim() || !activeImageSource;
    startCameraButton.disabled = hasVideo;
    startMicrophoneButton.disabled = hasAudio;
  }

  function getExplicitImageSourceSelection() {
    if (useCapturedImageRadio.checked) {
      return "captured";
    }
    if (useUploadedImageRadio.checked) {
      return "uploaded";
    }
    return null;
  }

  function refreshPreviewForSelectedImageSource() {
    if (activeImageSource === "captured" && capturedImageUrl) {
      imagePreview.src = capturedImageUrl;
      imageSourceLabel.textContent = "Captured frame";
      imagePreview.style.display = "block";
      clearCapturedFrameButton.style.display = "inline-flex";
    } else if (activeImageSource === "uploaded" && uploadedImageUrl) {
      imagePreview.src = uploadedImageUrl;
      imageSourceLabel.textContent = "Uploaded image";
      imagePreview.style.display = "block";
      clearCapturedFrameButton.style.display = "none";
    } else {
      imagePreview.src = "";
      imageSourceLabel.textContent = "No image selected";
      imagePreview.style.display = "none";
      clearCapturedFrameButton.style.display = "none";
    }
  }

  function syncImageSourceControls() {
    const hasCaptured = Boolean(capturedImageBytes);
    const hasUploaded = Boolean(uploadedImageBytes);
    useCapturedImageRadio.disabled = !hasCaptured;
    useUploadedImageRadio.disabled = !hasUploaded;

    const explicitSelection = getExplicitImageSourceSelection();
    if (hasCaptured && !hasUploaded) {
      activeImageSource = "captured";
      useCapturedImageRadio.checked = true;
    } else if (!hasCaptured && hasUploaded) {
      activeImageSource = "uploaded";
      useUploadedImageRadio.checked = true;
    } else if (hasCaptured && hasUploaded) {
      if (explicitSelection) {
        activeImageSource = explicitSelection;
      } else if (activeImageSource !== "captured" && activeImageSource !== "uploaded") {
        activeImageSource = "captured";
        useCapturedImageRadio.checked = true;
      }
    } else {
      activeImageSource = null;
      useCapturedImageRadio.checked = false;
      useUploadedImageRadio.checked = false;
    }

    refreshPreviewForSelectedImageSource();
  }

  function clearCapturedFrame() {
    if (capturedImageUrl) {
      URL.revokeObjectURL(capturedImageUrl);
    }
    capturedImageBytes = null;
    capturedImageUrl = null;

    if (uploadedImageBytes) {
      activeImageSource = "uploaded";
      useUploadedImageRadio.checked = true;
      useCapturedImageRadio.checked = false;
    } else {
      activeImageSource = null;
      useCapturedImageRadio.checked = false;
      useUploadedImageRadio.checked = false;
    }

    refreshPreviewForSelectedImageSource();
    updateButtonStates();
    setStatus("Captured frame canceled.");
  }

  function handleImageSourceSelectionChange() {
    if (useCapturedImageRadio.checked && capturedImageBytes) {
      activeImageSource = "captured";
    } else if (useUploadedImageRadio.checked && uploadedImageBytes) {
      activeImageSource = "uploaded";
    }
    refreshPreviewForSelectedImageSource();
    updateButtonStates();
  }

  async function startCamera() {
    console.log('startCamera invoked');
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      console.log('enumerateDevices:', devices);
      const hasVideoInput = devices.some((device) => device.kind === 'videoinput');
      if (!hasVideoInput) {
        setStatus('No camera device found.');
        updateButtonStates();
        return;
      }

      const videoStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: 'user',
        },
      });
      console.log('got videoStream tracks', videoStream.getVideoTracks().map((t) => `${t.kind}:${t.label}`));
      if (localStream) {
        videoStream.getVideoTracks().forEach((t) => localStream.addTrack(t));
      } else {
        localStream = videoStream;
      }
      camera.srcObject = localStream;
      camera.style.display = 'block';
      await camera.play().catch(() => {});
      setStatus('Camera started.');
    } catch (err) {
      console.error('startCamera failed', err);
      if (err.name === 'NotAllowedError') {
        setStatus('Camera access denied. Please allow camera permission.');
      } else if (err.name === 'NotFoundError') {
        setStatus('No camera found on this device.');
      } else {
        setStatus(`Could not start camera: ${err.message}`);
      }
    }
    updateButtonStates();
  }

  async function startMicrophone() {
    console.log('startMicrophone invoked');
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      console.log('enumerateDevices:', devices);
      const hasAudioInput = devices.some((device) => device.kind === 'audioinput');
      if (!hasAudioInput) {
        setStatus('No microphone device found.');
        updateButtonStates();
        return;
      }

      const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('got audioStream tracks', audioStream.getAudioTracks().map((t) => `${t.kind}:${t.label}`));
      if (localStream) {
        audioStream.getAudioTracks().forEach((t) => localStream.addTrack(t));
      } else {
        localStream = audioStream;
      }
      setStatus('Microphone started.');
    } catch (err) {
      console.error('startMicrophone failed', err);
      if (err.name === 'NotAllowedError') {
        setStatus('Microphone access denied. Please allow microphone permission.');
      } else if (err.name === 'NotFoundError') {
        setStatus('No microphone found on this device.');
      } else {
        setStatus(`Could not start microphone: ${err.message}`);
      }
    }
    updateButtonStates();
  }

  async function startSession() {
    try {
      localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
      camera.srcObject = localStream;
      setStatus("Camera and microphone connected.");
    } catch (error) {
      setStatus(`Could not start session: ${error.message}`);
      localStream = null;
    }
    updateButtonStates();
  }

  function stopCamera() {
    if (!localStream) {
      setStatus("No active session to stop camera.");
      return;
    }
    const videoTracks = localStream.getVideoTracks().filter((track) => track.readyState !== "ended");
    if (!videoTracks.length) {
      setStatus("Camera is already stopped.");
      return;
    }
    videoTracks.forEach((track) => {
      track.stop();
      localStream.removeTrack(track);
    });
    camera.srcObject = null;
    setStatus("Camera stopped.");
    if (!localStream.getTracks().length) {
      localStream = null;
    }
    updateButtonStates();
  }

  function stopMicrophone() {
    if (!localStream) {
      setStatus("No active session to stop microphone.");
      return;
    }
    const audioTracks = localStream.getAudioTracks().filter((track) => track.readyState !== "ended");
    if (!audioTracks.length) {
      setStatus("Microphone is already stopped.");
      return;
    }
    audioTracks.forEach((track) => {
      track.stop();
      localStream.removeTrack(track);
    });
    setStatus("Microphone stopped.");
    if (!localStream.getTracks().length) {
      localStream = null;
      camera.srcObject = null;
    }
    updateButtonStates();
  }

  function stopSession() {
    if (!localStream) {
      setStatus("No active session to stop.");
      return;
    }
    localStream.getTracks().forEach((track) => track.stop());
    localStream = null;
    camera.srcObject = null;
    setStatus("Session stopped.");
    updateButtonStates();
  }

  // Grabs a fresh frame from the live camera feed and returns a Blob.
  // Returns null if there's no live video to grab from.
  async function grabFrameBlob() {
    if (!localStream || !localStream.getVideoTracks().some((t) => t.readyState !== "ended")) {
      return null;
    }

    const track = localStream.getVideoTracks()[0];
    let blob;

    if (typeof ImageCapture === "function") {
      try {
        const imageCapture = new ImageCapture(track);
        const photo = await imageCapture.takePhoto();
        blob = new Blob([await photo.arrayBuffer()], { type: "image/jpeg" });
      } catch (error) {
        console.warn("ImageCapture failed", error);
      }
    }

    if (!blob) {
      if (!camera.videoWidth || !camera.videoHeight) {
        return null;
      }
      const canvas = document.createElement("canvas");
      canvas.width = camera.videoWidth;
      canvas.height = camera.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(camera, 0, 0, canvas.width, canvas.height);
      blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
      if (!blob) {
        return null;
      }
    }

    return blob;
  }

  function setCapturedImage(blob, arrayBuffer) {
    capturedImageBytes = arrayBuffer;
    if (capturedImageUrl) {
      URL.revokeObjectURL(capturedImageUrl);
    }
    capturedImageUrl = URL.createObjectURL(blob);
    activeImageSource = "captured";
    useCapturedImageRadio.checked = true;
    syncImageSourceControls();
  }

  async function captureFace() {
    if (!localStream || !localStream.getVideoTracks().length) {
      setStatus("Start the session before capturing the frame.");
      return;
    }

    const blob = await grabFrameBlob();
    if (!blob) {
      setStatus("Camera feed is not ready yet.");
      return;
    }

    const arrayBuffer = await blob.arrayBuffer();
    setCapturedImage(blob, arrayBuffer);
    setStatus("Frame extracted. Use Send Frame or Send Multimodal to send it.");
    updateButtonStates();
  }

  async function analyzeUploadedImage() {
    const file = imageUploadInput.files[0];
    if (!file) {
      setStatus("Select an image before analyzing.");
      return;
    }

    try {
      const arrayBuffer = uploadedImageBytes || (await file.arrayBuffer());
      uploadedImageBytes = arrayBuffer;
      if (!uploadedImageUrl) {
        uploadedImageUrl = URL.createObjectURL(file);
      }
      syncImageSourceControls();
      const response = await fetch("/api/image", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: arrayBuffer,
      });

      if (!response.ok) {
        const error = await response.text();
        console.error("analyzeUploadedImage error response:", error);
        setStatus(`Image analysis failed: ${error}`);
        return;
      }

      const data = await response.json();
      const meta = data.metadata || {};
      const filteredMeta = {
        source: meta.source,
        metrics: meta.metrics,
        text_emotion: meta.text_emotion,
        fused_emotion: meta.fused_emotion,
      };
      emotionMetadata.textContent = JSON.stringify(filteredMeta, null, 2);
      setStatus("Uploaded image analyzed.");
    } catch (error) {
      console.error("analyzeUploadedImage error:", error);
      setStatus(`Image analysis failed: ${error.message}`);
    }
    updateButtonStates();
  }

  async function updateUploadedImagePreview() {
    const file = imageUploadInput.files[0];
    if (!file) {
      uploadedImageBytes = null;
      if (uploadedImageUrl) {
        URL.revokeObjectURL(uploadedImageUrl);
        uploadedImageUrl = null;
      }
      syncImageSourceControls();
      updateButtonStates();
      return;
    }

    if (uploadedImageUrl) {
      URL.revokeObjectURL(uploadedImageUrl);
    }
    uploadedImageUrl = URL.createObjectURL(file);
    // Read the file immediately so it's usable (e.g. for Send Multimodal)
    // without requiring a separate "Analyze" click.
    uploadedImageBytes = await file.arrayBuffer();

    if (!capturedImageBytes && !activeImageSource) {
      activeImageSource = "uploaded";
      useUploadedImageRadio.checked = true;
    }

    if (activeImageSource === "uploaded") {
      imagePreview.src = uploadedImageUrl;
      imageSourceLabel.textContent = "Uploaded image";
      imagePreview.style.display = "block";
    }

    syncImageSourceControls();
    setStatus("Image selected. Click Analyze for emotion feedback, or send it directly with your message.");
    updateButtonStates();
  }

  // Send the previously extracted frame. If no frame has been extracted,
  // the user must click Extract Frame first.
  async function sendFrame() {
    if (!capturedImageBytes) {
      setStatus("Extract a frame before sending it.");
      return;
    }

    const arrayBuffer = capturedImageBytes;
    activeImageSource = "captured";

    try {
      const response = await fetch("/api/image", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: arrayBuffer,
      });

      if (!response.ok) {
        const error = await response.text();
        console.error("sendFrame error response:", error);
        setStatus(`Captured frame request failed: ${error}`);
        return;
      }

      const data = await response.json();
      const meta = data.metadata || {};
      const filteredMeta = {
        source: meta.source,
        metrics: meta.metrics,
        text_emotion: meta.text_emotion,
        fused_emotion: meta.fused_emotion,
      };
      emotionMetadata.textContent = JSON.stringify(filteredMeta, null, 2);
      setStatus("Captured frame sent and analyzed.");
    } catch (error) {
      console.error("sendFrame error:", error);
      setStatus(`Captured frame request failed: ${error.message}`);
    }
    updateButtonStates();
  }

  function createMediaRecorder() {
    const activeAudioTracks = localStream?.getAudioTracks().filter((track) => track.readyState !== "ended") || [];
    if (!activeAudioTracks.length) {
      return null;
    }
    const audioStream = new MediaStream([activeAudioTracks[0]]);
    try {
      let options = {};
      if (MediaRecorder.isTypeSupported) {
        if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
          options.mimeType = 'audio/webm;codecs=opus';
        } else if (MediaRecorder.isTypeSupported('audio/webm')) {
          options.mimeType = 'audio/webm';
        }
      }
      console.log('createMediaRecorder options:', options);
      const recorder = new MediaRecorder(audioStream, options);
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunks.push(event.data);
          console.log('ondataavailable: chunk size', event.data.size, 'audioChunks length', audioChunks.length);
          updateButtonStates();
        }
      };
      recorder.onstop = () => {
        console.log('recorder stopped, audioChunks length', audioChunks.length);
        setStatus("Voice recording stopped. Ready to send.");
        updateButtonStates();
      };
      recorder.onerror = (event) => {
        console.error("MediaRecorder error:", event.error);
        setStatus(`Recording error: ${event.error?.message || "Unknown"}`);
        updateButtonStates();
      };
      return recorder;
    } catch (error) {
      console.error("createMediaRecorder failed:", error);
      setStatus(`Could not create recorder: ${error.message}`);
      return null;
    }
  }

  async function startVoiceRecording() {
    if (!localStream || !localStream.getAudioTracks().length) {
      setStatus("Start the session before recording voice.");
      return;
    }
    audioChunks = [];
    mediaRecorder = createMediaRecorder();
    if (!mediaRecorder) {
      return;
    }
    try {
      mediaRecorder.start();
      console.log('mediaRecorder state after start:', mediaRecorder.state);
      setStatus("Recording voice...");
    } catch (err) {
      console.error('mediaRecorder.start() failed', err);
      setStatus(`Could not start recording: ${err.message}`);
      mediaRecorder = null;
    }
    updateButtonStates();
  }

  function stopVoiceRecording() {
    if (!mediaRecorder || mediaRecorder.state !== "recording") {
      setStatus("No active voice recording.");
      return;
    }
    try {
      mediaRecorder.stop();
      console.log('mediaRecorder stop called');
    } catch (err) {
      console.error('mediaRecorder.stop() failed', err);
      setStatus(`Could not stop recording: ${err.message}`);
    }
    updateButtonStates();
  }

  async function sendVoiceInput() {
    if (!audioChunks.length) {
      setStatus("Record audio before sending voice input.");
      return;
    }

    try {
      const mimeType = mediaRecorder?.mimeType || 'audio/webm';
      const blob = new Blob(audioChunks, { type: mimeType });
      console.log('sendVoiceInput: blob size', blob.size, 'mimeType', mimeType, 'audioChunks length', audioChunks.length);
      const arrayBuffer = await blob.arrayBuffer();
      const response = await fetch("/api/audio", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: arrayBuffer,
      });

      if (!response.ok) {
        const error = await response.text();
        console.error("sendVoiceInput error response:", error);
        setStatus(`Voice request failed: ${error}`);
        return;
      }

      const data = await response.json();
      voiceTranscript.textContent = data.transcript || "No transcript returned.";
      const fallbackMessage = "Sorry, I couldn't generate a response at the moment.";
      responseText.textContent = data.response && data.response !== fallbackMessage ? data.response : "";
      const meta = data.metadata || {};
      const filteredMeta = {
        source: meta.source,
        metrics: meta.metrics,
        text_emotion: meta.text_emotion,
        fused_emotion: meta.fused_emotion,
      };
      emotionMetadata.textContent = JSON.stringify(filteredMeta, null, 2);
      setStatus("Voice input processed.");
      audioChunks = [];
    } catch (error) {
      console.error("sendVoiceInput error:", error);
      setStatus(`Voice request failed: ${error.message}`);
    }
    updateButtonStates();
  }

  async function sendMultimodalInput() {
    if (!messageInput.value.trim()) {
      setStatus("Enter a message before sending multimodal input.");
      return;
    }
    if (!audioChunks.length) {
      setStatus("Record audio before sending multimodal input.");
      return;
    }
    const text = messageInput.value.trim();

    // Use whichever image source the user last interacted with. Both a
    // captured live frame and an uploaded image are treated identically here.
    let imageBytes = null;
    if (activeImageSource === "uploaded" && uploadedImageBytes) {
      imageBytes = uploadedImageBytes;
    } else if (activeImageSource === "captured" && capturedImageBytes) {
      imageBytes = capturedImageBytes;
    } else if (uploadedImageBytes) {
      imageBytes = uploadedImageBytes;
    } else if (capturedImageBytes) {
      imageBytes = capturedImageBytes;
    }

    // Nothing captured or uploaded yet, but the camera is live: grab a fresh
    // frame automatically rather than blocking the user.
    if (!imageBytes) {
      setStatus("Provide an extracted or uploaded image before sending multimodal input.");
      return;
    }

    try {
      const mimeType = mediaRecorder?.mimeType || 'audio/webm';
      const blob = new Blob(audioChunks, { type: mimeType });
      const formData = new FormData();
      formData.append('text', text);
      formData.append('image', new Blob([imageBytes], { type: 'image/jpeg' }), 'image.jpg');
      formData.append('audio', blob, 'audio.webm');

      const response = await fetch('/api/multimodal', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.text();
        console.error('sendMultimodalInput error response:', error);
        setStatus(`Multimodal request failed: ${error}`);
        return;
      }

      const data = await response.json();
      voiceTranscript.textContent = data.transcript || 'No transcript returned.';
      const fallbackMessage = "Sorry, I couldn't generate a response at the moment.";
      responseText.textContent = data.response && data.response !== fallbackMessage ? data.response : '';
      const meta = data.metadata || {};
      const filteredMeta = {
        source: meta.source,
        metrics: meta.metrics,
        text_emotion: meta.text_emotion,
        speech_sentiment: meta.speech_sentiment,
        face_emotion: meta.face_emotion,
        fused_emotion: meta.fused_emotion,
      };
      emotionMetadata.textContent = JSON.stringify(filteredMeta, null, 2);
      setStatus('Multimodal input processed.');
      audioChunks = [];
    } catch (error) {
      console.error('sendMultimodalInput error:', error);
      setStatus(`Multimodal request failed: ${error.message}`);
    }
    updateButtonStates();
  }

  async function sendTextMessage() {
    const text = messageInput.value.trim();
    if (!text) {
      setStatus("Enter a message before sending.");
      return;
    }
    sendTextButton.disabled = true;
    try {
      const response = await fetch("/api/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const error = await response.text();
        console.error("sendTextMessage error response:", error);
        setStatus(`Text request failed: ${error}`);
        return;
      }

      const data = await response.json();
      const fallbackMessage = "Sorry, I couldn't generate a response at the moment.";
      responseText.textContent = data.response && data.response !== fallbackMessage ? data.response : "";
      const meta = data.metadata || {};
      const filteredMeta = {
        source: meta.source,
        metrics: meta.metrics,
        text_emotion: meta.text_emotion,
        fused_emotion: meta.fused_emotion,
      };
      emotionMetadata.textContent = JSON.stringify(filteredMeta, null, 2);
      setStatus("Text sent and response generated.");
    } catch (error) {
      console.error("sendTextMessage error:", error);
      setStatus(`Text request failed: ${error.message}`);
    } finally {
      sendTextButton.disabled = false;
    }
  }

  startButton.addEventListener("click", startSession);
  startCameraButton.addEventListener("click", startCamera);
  startMicrophoneButton.addEventListener("click", startMicrophone);
  stopCameraButton.addEventListener("click", stopCamera);
  stopMicrophoneButton.addEventListener("click", stopMicrophone);
  stopSessionButton.addEventListener("click", stopSession);
  captureImageButton.addEventListener("click", captureFace);
  uploadImageButton.addEventListener("click", analyzeUploadedImage);
  sendFrameButton.addEventListener("click", sendFrame);
  clearCapturedFrameButton.addEventListener("click", clearCapturedFrame);
  sendTextButton.addEventListener("click", sendTextMessage);
  sendMultimodalButton.addEventListener("click", sendMultimodalInput);
  startRecordingButton.addEventListener("click", startVoiceRecording);
  stopRecordingButton.addEventListener("click", stopVoiceRecording);
  sendVoiceButton.addEventListener("click", sendVoiceInput);
  imageUploadInput.addEventListener("change", updateUploadedImagePreview);
  useCapturedImageRadio.addEventListener("change", handleImageSourceSelectionChange);
  useUploadedImageRadio.addEventListener("change", handleImageSourceSelectionChange);
  messageInput.addEventListener("input", updateButtonStates);

  updateButtonStates();
});