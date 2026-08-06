console.log("frontend app.js loaded");
document.addEventListener("DOMContentLoaded", () => {
  // ── DOM refs ─────────────────────────────────────────────────────────────
  const startButton             = document.getElementById("startButton");
  const stopCameraButton        = document.getElementById("stopCameraButton");
  const stopMicrophoneButton    = document.getElementById("stopMicrophoneButton");
  const stopSessionButton       = document.getElementById("stopSessionButton");
  const captureImageButton      = document.getElementById("captureImageButton");
  const uploadImageButton       = document.getElementById("uploadImageButton");
  const sendFrameButton         = document.getElementById("sendFrameButton");
  const sendTextButton          = document.getElementById("sendTextButton");
  const imageUploadInput        = document.getElementById("imageUploadInput");
  const useCapturedImageRadio   = document.getElementById("useCapturedImageRadio");
  const useUploadedImageRadio   = document.getElementById("useUploadedImageRadio");
  const statusMessage           = document.getElementById("statusMessage");
  const responseText            = document.getElementById("responseText");
  const emotionMetadata         = document.getElementById("emotionMetadata");
  const messageInput            = document.getElementById("messageInput");
  const voiceTranscript         = document.getElementById("voiceTranscript");
  const camera                  = document.getElementById("camera");
  const imagePreview            = document.getElementById("imagePreview");
  const imageSourceLabel        = document.getElementById("imageSourceLabel");
  const startRecordingButton    = document.getElementById("startRecordingButton");
  const stopRecordingButton     = document.getElementById("stopRecordingButton");
  const sendVoiceButton         = document.getElementById("sendVoiceButton");
  const sendMultimodalButton    = document.getElementById("sendMultimodalButton");
  const startCameraButton       = document.getElementById("startCameraButton");
  const startMicrophoneButton   = document.getElementById("startMicrophoneButton");
  const clearCapturedFrameButton= document.getElementById("clearCapturedFrameButton");
  const streamVideoBtn          = document.getElementById("streamVideoButton");
  const streamAudioBtn          = document.getElementById("streamAudioButton");

  // ── State ─────────────────────────────────────────────────────────────────
  let localStream        = null;
  let mediaRecorder      = null;
  let audioChunks        = [];
  let capturedImageBytes = null;
  let uploadedImageBytes = null;
  let capturedImageUrl   = null;
  let uploadedImageUrl   = null;
  let activeImageSource  = null; // 'captured' | 'uploaded' | null

  // WebSocket streaming state
  let videoWs             = null;
  let audioWs             = null;
  let videoStreamInterval = null;
  let audioStreamRecorder = null;
  let isStreamingVideo    = false;
  let isStreamingAudio    = false;

  // ── Helpers ───────────────────────────────────────────────────────────────
  function wsUrl(path) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}${path}`;
  }

  function setStatus(msg) { statusMessage.textContent = msg; }

  function updateButtonStates() {
    const hasStream   = Boolean(localStream);
    const hasVideo    = hasStream && localStream.getVideoTracks().some(t => t.readyState !== "ended");
    const hasAudio    = hasStream && localStream.getAudioTracks().some(t => t.readyState !== "ended");
    const isRecording = mediaRecorder?.state === "recording";

    stopCameraButton.disabled      = !hasVideo;
    stopMicrophoneButton.disabled  = !hasAudio;
    stopSessionButton.disabled     = !hasStream;
    captureImageButton.disabled    = !hasVideo;
    uploadImageButton.disabled     = !imageUploadInput.files.length;
    startRecordingButton.disabled  = !hasAudio || isRecording;
    stopRecordingButton.disabled   = !isRecording;
    sendVoiceButton.disabled       = isRecording || audioChunks.length === 0;
    sendFrameButton.disabled       = !capturedImageBytes;
    startCameraButton.disabled     = hasVideo;
    startMicrophoneButton.disabled = hasAudio;

    // Multimodal: text required + (local audio chunks OR live WS audio stream)
    //                           + (active image OR live WS video stream)
    const hasAudioForMultimodal = audioChunks.length > 0 || isStreamingAudio;
    const hasImageForMultimodal = Boolean(activeImageSource) || isStreamingVideo;
    sendMultimodalButton.disabled = isRecording || !messageInput.value.trim()
                                  || !hasAudioForMultimodal || !hasImageForMultimodal;

    if (streamVideoBtn) {
      streamVideoBtn.disabled     = !hasVideo;
      streamVideoBtn.textContent  = isStreamingVideo ? "Stop Video Stream" : "Stream Video";
      streamVideoBtn.classList.toggle("rail-btn-danger", isStreamingVideo);
    }
    if (streamAudioBtn) {
      streamAudioBtn.disabled    = !hasAudio;
      streamAudioBtn.textContent = isStreamingAudio ? "Stop Audio Stream" : "Stream Audio";
      streamAudioBtn.classList.toggle("rail-btn-danger", isStreamingAudio);
    }
  }

  // ── WebSocket: video streaming ────────────────────────────────────────────
  // Sends JPEG frames at ~400 ms cadence. Receives face_emotion updates and
  // merges them into the live emotion display. Server stores latest face state
  // so /api/multimodal always has an up-to-date face reading.
  function startVideoStream() {
    if (isStreamingVideo) { stopVideoStream(); return; }
    if (!localStream || !localStream.getVideoTracks().some(t => t.readyState !== "ended")) {
      setStatus("Start the camera before streaming video."); return;
    }

    videoWs = new WebSocket(wsUrl("/ws/video"));
    videoWs.binaryType = "arraybuffer";

    videoWs.onopen = () => {
      isStreamingVideo = true;
      setStatus("Video stream active — face emotion updating live.");
      updateButtonStates();

      // 400 ms cadence (within 300–500 ms spec)
      videoStreamInterval = setInterval(async () => {
        if (videoWs?.readyState !== WebSocket.OPEN) return;
        const blob = await grabFrameBlob();
        if (!blob) return;
        videoWs.send(await blob.arrayBuffer());
      }, 400);
    };

    videoWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.face_emotion) {
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

  // ── WebSocket: audio streaming ────────────────────────────────────────────
  // Sends ~750 ms MediaRecorder timeslices to the server. Server transcribes
  // each chunk via Deepgram, classifies speech emotion, and stores the latest
  // state for /api/multimodal. Does NOT generate AI responses here — only
  // updates the live transcript and speech emotion display.
  function startAudioStream() {
    if (isStreamingAudio) { stopAudioStream(); return; }
    const audioTracks = localStream?.getAudioTracks().filter(t => t.readyState !== "ended") || [];
    if (!audioTracks.length) {
      setStatus("Start the microphone before streaming audio."); return;
    }

    audioWs = new WebSocket(wsUrl("/ws/audio"));
    audioWs.binaryType = "arraybuffer";

    audioWs.onopen = () => {
      isStreamingAudio = true;
      setStatus("Audio stream active — transcript updating live.");
      updateButtonStates();

      const audioStream = new MediaStream([audioTracks[0]]);
      let opts = {};
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) opts.mimeType = "audio/webm;codecs=opus";
      else if (MediaRecorder.isTypeSupported("audio/webm"))        opts.mimeType = "audio/webm";

      audioStreamRecorder = new MediaRecorder(audioStream, opts);
      audioStreamRecorder.ondataavailable = async (e) => {
        if (!e.data || e.data.size === 0 || audioWs?.readyState !== WebSocket.OPEN) return;
        audioWs.send(await e.data.arrayBuffer());
      };
      audioStreamRecorder.start(750); // 750 ms timeslice (500–1000 ms range)
    };

    // Only updates transcript + speech emotion — no AI response here
    audioWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.transcript) voiceTranscript.textContent = data.transcript;
        if (data.speech_emotion) {
          let current = {};
          try { current = JSON.parse(emotionMetadata.textContent); } catch (_) {}
          current.speech_emotion = data.speech_emotion;
          if (data.transcript) current.live_transcript = data.transcript;
          emotionMetadata.textContent = JSON.stringify(current, null, 2);
        }
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

  // ── Image source controls ─────────────────────────────────────────────────
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

    const explicit = getExplicitImageSourceSelection();
    if (hasCaptured && !hasUploaded) {
      activeImageSource = "captured"; useCapturedImageRadio.checked = true;
    } else if (!hasCaptured && hasUploaded) {
      activeImageSource = "uploaded"; useUploadedImageRadio.checked = true;
    } else if (hasCaptured && hasUploaded) {
      if (explicit) activeImageSource = explicit;
      else if (activeImageSource !== "captured" && activeImageSource !== "uploaded") {
        activeImageSource = "captured"; useCapturedImageRadio.checked = true;
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
    setStatus("Captured frame cleared.");
  }

  function handleImageSourceSelectionChange() {
    if (useCapturedImageRadio.checked && capturedImageBytes) activeImageSource = "captured";
    else if (useUploadedImageRadio.checked && uploadedImageBytes) activeImageSource = "uploaded";
    refreshPreviewForSelectedImageSource();
    updateButtonStates();
  }

  // ── Camera / microphone controls ──────────────────────────────────────────
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
      if (err.name === "NotAllowedError") setStatus("Camera access denied. Please allow camera permission.");
      else if (err.name === "NotFoundError") setStatus("No camera found on this device.");
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
      if (err.name === "NotAllowedError") setStatus("Microphone access denied. Please allow microphone permission.");
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
    } catch (err) {
      setStatus(`Could not start session: ${err.message}`);
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

  // ── Frame capture ─────────────────────────────────────────────────────────
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

  // ── HTTP API calls ────────────────────────────────────────────────────────
  async function analyzeUploadedImage() {
    const file = imageUploadInput.files[0];
    if (!file) { setStatus("Select an image before analyzing."); return; }
    try {
      uploadedImageBytes = uploadedImageBytes || await file.arrayBuffer();
      if (!uploadedImageUrl) uploadedImageUrl = URL.createObjectURL(file);
      syncImageSourceControls();
      const res = await fetch("/api/image", {
        method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: uploadedImageBytes,
      });
      if (!res.ok) { setStatus(`Image analysis failed: ${await res.text()}`); return; }
      const data = await res.json();
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify(
        { source: meta.source, text_emotion: meta.text_emotion, fused_emotion: meta.fused_emotion }, null, 2);
      setStatus("Uploaded image analyzed.");
    } catch (err) { setStatus(`Image analysis failed: ${err.message}`); }
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
      const res = await fetch("/api/image", {
        method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: capturedImageBytes,
      });
      if (!res.ok) { setStatus(`Frame request failed: ${await res.text()}`); return; }
      const data = await res.json();
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify(
        { source: meta.source, text_emotion: meta.text_emotion, fused_emotion: meta.fused_emotion }, null, 2);
      setStatus("Captured frame sent and analyzed.");
    } catch (err) { setStatus(`Frame request failed: ${err.message}`); }
    updateButtonStates();
  }

  // ── Manual voice recording (HTTP /api/audio) ──────────────────────────────
  function createMediaRecorder() {
    const tracks = localStream?.getAudioTracks().filter(t => t.readyState !== "ended") || [];
    if (!tracks.length) return null;
    const audioStream = new MediaStream([tracks[0]]);
    let opts = {};
    if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) opts.mimeType = "audio/webm;codecs=opus";
    else if (MediaRecorder.isTypeSupported("audio/webm"))        opts.mimeType = "audio/webm";
    try {
      const recorder = new MediaRecorder(audioStream, opts);
      recorder.ondataavailable = (e) => { if (e.data?.size > 0) { audioChunks.push(e.data); updateButtonStates(); } };
      recorder.onstop  = () => { setStatus("Recording stopped. Ready to send."); updateButtonStates(); };
      recorder.onerror = (e) => setStatus(`Recording error: ${e.error?.message || "Unknown"}`);
      return recorder;
    } catch (err) { setStatus(`Could not create recorder: ${err.message}`); return null; }
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
      const blob = new Blob(audioChunks, { type: mediaRecorder?.mimeType || "audio/webm" });
      const res = await fetch("/api/audio", {
        method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: await blob.arrayBuffer(),
      });
      if (!res.ok) { setStatus(`Voice request failed: ${await res.text()}`); return; }
      const data = await res.json();
      voiceTranscript.textContent = data.transcript || "No transcript returned.";
      responseText.textContent = data.response || "";
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify(
        { source: meta.source, text_emotion: meta.text_emotion, fused_emotion: meta.fused_emotion }, null, 2);
      setStatus("Voice input processed.");
      audioChunks = [];
    } catch (err) { setStatus(`Voice request failed: ${err.message}`); }
    updateButtonStates();
  }

  // ── Multimodal (HTTP /api/multimodal) ─────────────────────────────────────
  // If WS streams are active the server already has fresh face/speech state,
  // so image and audio blobs are optional — only text is required.
  async function sendMultimodalInput() {
    const text = messageInput.value.trim();
    if (!text) { setStatus("Enter a message first."); return; }

    const formData = new FormData();
    formData.append("text", text);

    // Attach image blob if available locally; otherwise server uses WS state
    let imageBytes = activeImageSource === "uploaded" ? uploadedImageBytes
                   : activeImageSource === "captured" ? capturedImageBytes
                   : uploadedImageBytes || capturedImageBytes;
    if (imageBytes) formData.append("image", new Blob([imageBytes], { type: "image/jpeg" }), "image.jpg");

    // Attach audio blob if local chunks exist; otherwise server uses WS state
    if (audioChunks.length) {
      const blob = new Blob(audioChunks, { type: mediaRecorder?.mimeType || "audio/webm" });
      formData.append("audio", blob, "audio.webm");
    }

    try {
      const res = await fetch("/api/multimodal", { method: "POST", body: formData });
      if (!res.ok) { setStatus(`Multimodal request failed: ${await res.text()}`); return; }
      const data = await res.json();
      voiceTranscript.textContent = data.transcript || "No transcript returned.";
      responseText.textContent = data.response || "";
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify({
        source: meta.source, face_emotion: meta.face_emotion,
        speech_sentiment: meta.speech_sentiment, text_emotion: meta.text_emotion,
        fused_emotion: meta.fused_emotion,
      }, null, 2);
      setStatus("Multimodal input processed.");
      audioChunks = [];
    } catch (err) { setStatus(`Multimodal request failed: ${err.message}`); }
    updateButtonStates();
  }

  // ── Text only ─────────────────────────────────────────────────────────────
  async function sendTextMessage() {
    const text = messageInput.value.trim();
    if (!text) { setStatus("Enter a message first."); return; }
    sendTextButton.disabled = true;
    try {
      const res = await fetch("/api/text", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
      });
      if (!res.ok) { setStatus(`Text request failed: ${await res.text()}`); return; }
      const data = await res.json();
      responseText.textContent = data.response || "";
      const meta = data.metadata || {};
      emotionMetadata.textContent = JSON.stringify(
        { source: meta.source, text_emotion: meta.text_emotion, fused_emotion: meta.fused_emotion }, null, 2);
      setStatus("Text sent and response generated.");
    } catch (err) { setStatus(`Text request failed: ${err.message}`); }
    finally { sendTextButton.disabled = false; }
  }

  // ── Event listeners ───────────────────────────────────────────────────────
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
