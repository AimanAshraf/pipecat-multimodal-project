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