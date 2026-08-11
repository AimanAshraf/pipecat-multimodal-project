console.log("frontend app.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // DOM REFERENCES
  // =========================================================================

  const startButton              = document.getElementById("startButton");
  const stopCameraButton         = document.getElementById("stopCameraButton");
  const stopMicrophoneButton     = document.getElementById("stopMicrophoneButton");
  const stopSessionButton        = document.getElementById("stopSessionButton");
  const captureImageButton       = document.getElementById("captureImageButton");
  const uploadImageButton        = document.getElementById("uploadImageButton");
  const sendFrameButton          = document.getElementById("sendFrameButton");
  const sendTextButton           = document.getElementById("sendTextButton");
  const imageUploadInput         = document.getElementById("imageUploadInput");
  const useCapturedImageRadio    = document.getElementById("useCapturedImageRadio");
  const useUploadedImageRadio    = document.getElementById("useUploadedImageRadio");
  const statusMessage            = document.getElementById("statusMessage");
  const responseText             = document.getElementById("responseText");
  const emotionMetadata          = document.getElementById("emotionMetadata");
  const messageInput             = document.getElementById("messageInput");
  const voiceTranscript          = document.getElementById("voiceTranscript");
  const camera                   = document.getElementById("camera");
  const imagePreview             = document.getElementById("imagePreview");
  const imageSourceLabel         = document.getElementById("imageSourceLabel");
  const startRecordingButton     = document.getElementById("startRecordingButton");
  const stopRecordingButton      = document.getElementById("stopRecordingButton");
  const sendVoiceButton          = document.getElementById("sendVoiceButton");
  const sendMultimodalButton     = document.getElementById("sendMultimodalButton");
  const startCameraButton        = document.getElementById("startCameraButton");
  const startMicrophoneButton    = document.getElementById("startMicrophoneButton");
  const clearCapturedFrameButton = document.getElementById("clearCapturedFrameButton");

  const streamVideoBtn           = document.getElementById("streamVideoButton");
  const streamAudioBtn           = document.getElementById("streamAudioButton");
  const controlRail              = document.getElementById("controlRail");
  const modeManualButton         = document.getElementById("modeManualButton");
  const modeLiveButton           = document.getElementById("modeLiveButton");


  // =========================================================================
  // STATE
  // =========================================================================

  let localStream = null;

  // Manual recording
  let mediaRecorder = null;
  let audioChunks = [];

  // Image state
  let capturedImageBytes = null;
  let uploadedImageBytes = null;

  let capturedImageUrl = null;
  let uploadedImageUrl = null;

  let activeImageSource = null;


  // =========================================================================
  // LIVE STREAMING STATE
  // =========================================================================

  let videoWs = null;
  let audioWs = null;

  let isStreamingVideo = false;
  let isStreamingAudio = false;

  // The new architecture uses a 4-second analysis window.
  const ANALYSIS_INTERVAL_MS = 4000;

  // Video frames are sampled more frequently than the analysis interval.
  const VIDEO_SAMPLE_INTERVAL_MS = 400;

  // Audio chunks are small so transcript can be updated continuously.
  const AUDIO_CHUNK_INTERVAL_MS = 750;

  let videoSampleInterval = null;

  let audioStreamRecorder = null;

  // Indicates whether the browser is currently collecting
  // data for the current 2-second analysis window.
  let analysisWindowActive = false;

  // Number of video frames sent during the current window.
  let videoFramesSentInWindow = 0;

  // Number of audio chunks sent during the current window.
  let audioChunksSentInWindow = 0;

  // Used to show which 2-second window is currently being processed.
  let analysisWindowNumber = 0;

  // Prevents overlapping analysis windows.
  let analysisInProgress = false;


  // =========================================================================
  // HELPERS
  // =========================================================================

  function wsUrl(path) {
    const protocol =
      location.protocol === "https:" ? "wss" : "ws";

    return `${protocol}://${location.host}${path}`;
  }


  function setStatus(message) {
    if (statusMessage) {
      statusMessage.textContent = message;
    }
  }


  function updateEmotionMetadata(updates) {
    let current = {};

    try {
      const existing = emotionMetadata.textContent.trim();

      if (
        existing &&
        existing !== "No data yet." &&
        existing.startsWith("{")
      ) {
        current = JSON.parse(existing);
      }
    } catch (_) {
      current = {};
    }

    // Merge updates, but remove keys with null/undefined values
    for (const [key, value] of Object.entries(updates)) {
      if (value === null || value === undefined) {
        delete current[key];
      } else {
        current[key] = value;
      }
    }

    // Rebuild object in the desired order:
    // 1. face_emotion (if present AND video is streaming)
    // 2. speech_emotion (if present AND audio is streaming)
    // 3. fused_emotion (if present)
    // 4. everything else
    const ordered = {};

    // Only show face_emotion if video is actively streaming
    if (current.face_emotion && isStreamingVideo) {
      ordered.face_emotion = current.face_emotion;
    }

    // Only show speech_emotion if audio is actively streaming
    if (current.speech_emotion && isStreamingAudio) {
      ordered.speech_emotion = current.speech_emotion;
    }

    if (current.fused_emotion) {
      ordered.fused_emotion = current.fused_emotion;
    }

    // Add all other fields (skip the three handled above)
    for (const [key, value] of Object.entries(current)) {
      if (key !== 'face_emotion' && key !== 'speech_emotion' && key !== 'fused_emotion') {
        ordered[key] = value;
      }
    }

    emotionMetadata.textContent =
      JSON.stringify(ordered, null, 2);
  }


  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }


  // Renders the transcript text into #voiceTranscript, and — if the
  // given speechEmotion object carries a `snippet` field naming the
  // substring that most informed the emotion call — wraps that
  // substring in a highlighted <mark> so it's visually called out.
  // Falls back to plain text whenever there's no snippet, or the
  // snippet doesn't actually appear in the transcript verbatim.
  function renderVoiceTranscript(transcriptText, speechEmotion) {
    const text = transcriptText || "No voice recorded yet.";
    const snippet =
      speechEmotion && typeof speechEmotion.snippet === "string"
        ? speechEmotion.snippet.trim()
        : "";

    if (!snippet) {
      voiceTranscript.textContent = text;
      return;
    }

    // Case-insensitive search so minor casing differences from the
    // LLM's returned snippet don't silently fail to highlight.
    const lowerText = text.toLowerCase();
    const lowerSnippet = snippet.toLowerCase();
    const startIndex = lowerText.indexOf(lowerSnippet);

    if (startIndex === -1) {
      // Snippet doesn't match the transcript verbatim — degrade
      // gracefully to plain text rather than showing nothing.
      voiceTranscript.textContent = text;
      return;
    }

    const endIndex = startIndex + snippet.length;

    const before = escapeHtml(text.slice(0, startIndex));
    const matched = escapeHtml(text.slice(startIndex, endIndex));
    const after = escapeHtml(text.slice(endIndex));

    const emotionLabel = speechEmotion.emotion
      ? escapeHtml(speechEmotion.emotion)
      : "";

    voiceTranscript.innerHTML =
      `${before}<mark class="emotion-highlight" title="Drove the '${emotionLabel}' speech-emotion reading">${matched}</mark>${after}`;
  }


  function updateButtonStates() {
    const hasStream = Boolean(localStream);

    const hasVideo =
      hasStream &&
      localStream
        .getVideoTracks()
        .some(track => track.readyState !== "ended");

    const hasAudio =
      hasStream &&
      localStream
        .getAudioTracks()
        .some(track => track.readyState !== "ended");

    const isRecording =
      mediaRecorder?.state === "recording";


    if (stopCameraButton) {
      stopCameraButton.disabled = !hasVideo;
    }

    if (stopMicrophoneButton) {
      stopMicrophoneButton.disabled = !hasAudio;
    }

    if (stopSessionButton) {
      stopSessionButton.disabled = !hasStream;
    }

    if (captureImageButton) {
      captureImageButton.disabled = !hasVideo;
    }

    if (uploadImageButton) {
      uploadImageButton.disabled =
        !imageUploadInput?.files.length;
    }

    if (startRecordingButton) {
      startRecordingButton.disabled =
        !hasAudio || isRecording;
    }

    if (stopRecordingButton) {
      stopRecordingButton.disabled =
        !isRecording;
    }

    if (sendVoiceButton) {
      sendVoiceButton.disabled =
        isRecording || audioChunks.length === 0;
    }

    if (sendFrameButton) {
      sendFrameButton.disabled =
        !capturedImageBytes;
    }

    if (startCameraButton) {
      startCameraButton.disabled = hasVideo;
    }

    if (startMicrophoneButton) {
      startMicrophoneButton.disabled = hasAudio;
    }


    const hasAudioForMultimodal =
      audioChunks.length > 0 ||
      isStreamingAudio;

    const hasImageForMultimodal =
      Boolean(activeImageSource) ||
      isStreamingVideo;


    if (sendMultimodalButton) {
      sendMultimodalButton.disabled =
        isRecording ||
        !messageInput.value.trim() ||
        !hasAudioForMultimodal ||
        !hasImageForMultimodal;
    }


    if (streamVideoBtn) {
      streamVideoBtn.disabled = !hasVideo;

      streamVideoBtn.textContent =
        isStreamingVideo
          ? "Stop Video Stream"
          : "Stream Video";

      streamVideoBtn.classList.toggle(
        "rail-btn-danger",
        isStreamingVideo
      );
    }


    if (streamAudioBtn) {
      streamAudioBtn.disabled = !hasAudio;

      streamAudioBtn.textContent =
        isStreamingAudio
          ? "Stop Audio Stream"
          : "Stream Audio";

      streamAudioBtn.classList.toggle(
        "rail-btn-danger",
        isStreamingAudio
      );
    }
  }


  // =========================================================================
  // VIDEO FRAME CAPTURE
  // =========================================================================

  async function grabFrameBlob() {
    if (
      !localStream ||
      !localStream
        .getVideoTracks()
        .some(track => track.readyState !== "ended")
    ) {
      return null;
    }


    const track =
      localStream
        .getVideoTracks()
        .find(track => track.readyState !== "ended");


    let blob = null;


    // Try ImageCapture first.
    if (typeof ImageCapture === "function") {
      try {
        const imageCapture =
          new ImageCapture(track);

        const photo =
          await imageCapture.takePhoto();

        blob = new Blob(
          [await photo.arrayBuffer()],
          {
            type: "image/jpeg"
          }
        );
      } catch (_) {
        blob = null;
      }
    }


    // Fallback to video element + canvas.
    if (!blob) {
      if (
        !camera.videoWidth ||
        !camera.videoHeight
      ) {
        return null;
      }


      const canvas =
        document.createElement("canvas");

      canvas.width = camera.videoWidth;
      canvas.height = camera.videoHeight;


      const context =
        canvas.getContext("2d");

      if (!context) {
        return null;
      }


      context.drawImage(
        camera,
        0,
        0,
        canvas.width,
        canvas.height
      );


      blob =
        await new Promise(resolve => {
          canvas.toBlob(
            resolve,
            "image/jpeg",
            0.85
          );
        });
    }


    return blob || null;
  }


  // =========================================================================
  // 2-SECOND VIDEO STREAM
  // =========================================================================
  //
  // Important:
  //
  // The browser captures frames continuously.
  //
  //      frame -> frame -> frame -> frame
  //                 |
  //                 v
  //             2 seconds
  //                 |
  //                 v
  //          backend analysis
  //
  // We DO NOT call the LLM for every 400ms frame.
  //
  // The backend receives the frames and performs the actual
  // face + audio fusion when the 2-second window completes.
  //
  // =========================================================================

  function startVideoStream() {
    if (isStreamingVideo) {
      stopVideoStream();
      return;
    }


    if (
      !localStream ||
      !localStream
        .getVideoTracks()
        .some(track => track.readyState !== "ended")
    ) {
      setStatus(
        "Start the camera before streaming video."
      );

      return;
    }


    videoWs =
      new WebSocket(
        wsUrl("/ws/video")
      );

    videoWs.binaryType =
      "arraybuffer";


    videoWs.onopen = () => {
      isStreamingVideo = true;

      analysisWindowActive = true;

      videoFramesSentInWindow = 0;

      analysisWindowNumber = 0;

      setStatus(
        "Live video connected. Analyzing every 4 seconds..."
      );

      console.log("[MEDIA] Video WebSocket connected");

      updateButtonStates();


      // ---------------------------------------------------------
      // Sample video frames every 400 ms.
      // ---------------------------------------------------------

      videoSampleInterval =
        setInterval(
          async () => {

            if (
              !videoWs ||
              videoWs.readyState !== WebSocket.OPEN
            ) {
              return;
            }


            const blob =
              await grabFrameBlob();


            if (!blob) {
              return;
            }


            try {
              videoWs.send(
                await blob.arrayBuffer()
              );

              videoFramesSentInWindow++;

              if (videoFramesSentInWindow === 1) {
                console.log("[MEDIA] First video frame sent");
              }

            } catch (error) {
              console.error(
                "Video frame send failed:",
                error
              );
            }

          },
          VIDEO_SAMPLE_INTERVAL_MS
        );
    };


    videoWs.onmessage = event => {
      try {
        const data =
          JSON.parse(event.data);

        console.log("[VIDEO WS] Received:", data);

        // -------------------------------------------------------
        // 4-second fused result with AI response
        // -------------------------------------------------------

        if (data.type === "live_emotion" || data.type === "emotion_result") {

          analysisInProgress = false;


          const result =
            data.result || {};


          const faceEmotion =
            result.face_emotion ||
            data.face_emotion ||
            null;


          const speechEmotion =
            result.speech_emotion ||
            data.speech_emotion ||
            data.speech_sentiment ||
            null;


          const fusedEmotion =
            result.fused_emotion ||
            data.fused_emotion ||
            null;

          
          const aiResponse =
            data.ai_response ||
            result.ai_response ||
            "";

          console.log("[VIDEO WS] AI Response:", aiResponse);


          updateEmotionMetadata({


            face_emotion:
              faceEmotion,

            speech_emotion:
              speechEmotion,

            fused_emotion:
              fusedEmotion,

            analysis_window:
              data.window ||
              data.window_id ||
              analysisWindowNumber,

            interval_ms:
              ANALYSIS_INTERVAL_MS,


            video_frames:
              data.video_frames ||
              videoFramesSentInWindow,

            audio_chunks:
              data.audio_chunks ||
              audioChunksSentInWindow
          });


          // Display the AI response prominently
          if (aiResponse && aiResponse.trim()) {
            responseText.textContent = aiResponse;
            console.log("[AI] Response displayed: " + aiResponse);
          } else {
            console.log("[AI] No response in this message");
          }


          // Display the dominant fused emotion prominently
          // in the status message.

          if (fusedEmotion?.emotion) {

            const confidence =
              fusedEmotion.confidence != null
                ? Number(
                    fusedEmotion.confidence
                  ).toFixed(2)
                : null;


            setStatus(
              confidence
                ? `Emotion: ${fusedEmotion.emotion} (${confidence})`
                : `Emotion: ${fusedEmotion.emotion}`
            );
            
            console.log("[FUSION] Result: " + fusedEmotion.emotion + " (" + confidence + ")");
          }
        }


        // -------------------------------------------------------
        // Backward compatibility:
        // old backend may still send face_emotion.
        // -------------------------------------------------------

        else if (data.face_emotion) {

          updateEmotionMetadata({
            face_emotion:
              data.face_emotion
          });
        }


        // -------------------------------------------------------
        // Live transcript can also arrive through video WS.
        // -------------------------------------------------------

        if (data.transcript) {

          voiceTranscript.textContent =
            data.transcript;
        }


        if (data.speech_emotion) {

          updateEmotionMetadata({
            speech_emotion:
              data.speech_emotion
          });
        }

      } catch (error) {

        console.error(
          "Could not process video WS message:",
          error
        );
      }
    };


    videoWs.onerror = error => {

      console.error(
        "Video WebSocket error:",
        error
      );

      setStatus(
        "Video WebSocket error."
      );
    };


    videoWs.onclose = () => {

      isStreamingVideo = false;

      clearInterval(
        videoSampleInterval
      );

      videoSampleInterval = null;

      analysisWindowActive = false;

      updateButtonStates();
    };
  }


  function stopVideoStream() {

    clearInterval(
      videoSampleInterval
    );

    videoSampleInterval = null;


    if (videoWs) {

      try {
        videoWs.close();
      } catch (_) {}

      videoWs = null;
    }


    isStreamingVideo = false;

    analysisWindowActive = false;

    setStatus(
      "Video stream stopped."
    );

    // Immediately refresh metadata display to remove face_emotion
    // Call updateEmotionMetadata with empty object to trigger re-ordering
    updateEmotionMetadata({});

    updateButtonStates();
  }


  // =========================================================================
  // AUDIO STREAM
  // =========================================================================
  //
  // Audio is intentionally sent in smaller chunks.
  //
  // This allows Deepgram to continuously produce transcript updates.
  //
  // Example:
  //
  // 0.0s ---- 0.75s ---- 1.5s ---- 2.0s
  //       chunk       chunk
  //
  // The backend keeps the audio/emotion state and uses it when
  // the 2-second video analysis window completes.
  //
  // =========================================================================

  function startAudioStream() {

    if (isStreamingAudio) {
      stopAudioStream();
      return;
    }


    const audioTracks =
      localStream
        ?.getAudioTracks()
        .filter(
          track =>
            track.readyState !== "ended"
        ) || [];


    if (!audioTracks.length) {

      setStatus(
        "Start the microphone before streaming audio."
      );

      return;
    }


    audioWs =
      new WebSocket(
        wsUrl("/ws/audio")
      );

    audioWs.binaryType =
      "arraybuffer";


    audioWs.onopen = () => {

      isStreamingAudio = true;

      audioChunksSentInWindow = 0;

      setStatus(
        "Live audio connected. Transcription active."
      );

      console.log("[MEDIA] Audio WebSocket connected");

      updateButtonStates();


      const audioStream =
        new MediaStream([
          audioTracks[0]
        ]);


      const options = {};


      if (
        MediaRecorder.isTypeSupported(
          "audio/webm;codecs=opus"
        )
      ) {

        options.mimeType =
          "audio/webm;codecs=opus";

      } else if (
        MediaRecorder.isTypeSupported(
          "audio/webm"
        )
      ) {

        options.mimeType =
          "audio/webm";
      }


      try {

        audioStreamRecorder =
          new MediaRecorder(
            audioStream,
            options
          );

      } catch (error) {

        console.error(
          "MediaRecorder creation failed:",
          error
        );

        setStatus(
          `Could not create audio recorder: ${error.message}`
        );

        return;
      }


      audioStreamRecorder.ondataavailable =
        async event => {

          if (
            !event.data ||
            event.data.size === 0 ||
            !audioWs ||
            audioWs.readyState !== WebSocket.OPEN
          ) {
            return;
          }


          try {

            audioWs.send(
              await event.data.arrayBuffer()
            );

            audioChunksSentInWindow++;

            if (audioChunksSentInWindow === 1) {
              console.log("[MEDIA] First audio chunk sent");
            }

          } catch (error) {

            console.error(
              "Audio chunk send failed:",
              error
            );
          }
        };


      audioStreamRecorder.onerror =
        event => {

          console.error(
            "Audio recorder error:",
            event
          );
        };


      audioStreamRecorder.start(
        AUDIO_CHUNK_INTERVAL_MS
      );
    };


    // -----------------------------------------------------------
    // Deepgram transcript messages
    // -----------------------------------------------------------

    audioWs.onmessage = event => {

      try {

        const data =
          JSON.parse(event.data);


        // Real-time transcript
        if (data.transcript) {

          voiceTranscript.textContent =
            data.transcript;
          
          console.log("[STT] Transcript: " + data.transcript);
        }


        // Speech emotion
        if (data.speech_emotion) {

          updateEmotionMetadata({
            speech_emotion:
              data.speech_emotion
          });
        }


        // Some backend versions use speech_sentiment.
        if (data.speech_sentiment) {

          updateEmotionMetadata({
            speech_emotion:
              data.speech_sentiment
          });
        }


        // If backend sends a fused result through audio WS,
        // display it too.

        if (data.fused_emotion) {

          updateEmotionMetadata({
            fused_emotion:
              data.fused_emotion
          });
        }

        // Display AI response if provided
        if (data.ai_response) {
          responseText.textContent = data.ai_response;
          console.log("[AI] Response: " + data.ai_response);
        }

        // Handle live_emotion type
        if (data.type === "live_emotion") {
          if (data.ai_response) {
            responseText.textContent = data.ai_response;
          }
          
          if (data.fused_emotion) {
            updateEmotionMetadata({
              fused_emotion: data.fused_emotion,
              face_emotion: data.face_emotion,
              speech_emotion: data.speech_emotion
            });
          }
        }

      } catch (error) {

        console.error(
          "Could not process audio WS message:",
          error
        );
      }
    };


    audioWs.onerror = error => {

      console.error(
        "Audio WebSocket error:",
        error
      );

      setStatus(
        "Audio WebSocket error."
      );
    };


    audioWs.onclose = () => {

      isStreamingAudio = false;


      if (
        audioStreamRecorder &&
        audioStreamRecorder.state === "recording"
      ) {

        try {
          audioStreamRecorder.stop();
        } catch (_) {}
      }


      audioStreamRecorder = null;

      updateButtonStates();
    };
  }


  function stopAudioStream() {

    if (
      audioStreamRecorder &&
      audioStreamRecorder.state === "recording"
    ) {

      try {
        audioStreamRecorder.stop();
      } catch (_) {}
    }


    audioStreamRecorder = null;


    if (audioWs) {

      try {
        audioWs.close();
      } catch (_) {}

      audioWs = null;
    }


    isStreamingAudio = false;

    setStatus(
      "Audio stream stopped."
    );

    // Immediately refresh metadata display to remove speech_emotion
    // Call updateEmotionMetadata with empty object to trigger re-ordering
    updateEmotionMetadata({});

    updateButtonStates();
  }


  // =========================================================================
  // IMAGE SOURCE CONTROLS
  // =========================================================================

  function getExplicitImageSourceSelection() {

    if (
      useCapturedImageRadio?.checked
    ) {
      return "captured";
    }


    if (
      useUploadedImageRadio?.checked
    ) {
      return "uploaded";
    }


    return null;
  }


  function refreshPreviewForSelectedImageSource() {

    if (
      activeImageSource === "captured" &&
      capturedImageUrl
    ) {

      imagePreview.src =
        capturedImageUrl;

      imageSourceLabel.textContent =
        "Captured frame";

      imagePreview.style.display =
        "block";

      clearCapturedFrameButton.style.display =
        "inline-flex";

    } else if (
      activeImageSource === "uploaded" &&
      uploadedImageUrl
    ) {

      imagePreview.src =
        uploadedImageUrl;

      imageSourceLabel.textContent =
        "Uploaded image";

      imagePreview.style.display =
        "block";

      clearCapturedFrameButton.style.display =
        "none";

    } else {

      imagePreview.src = "";

      imageSourceLabel.textContent =
        "No image selected";

      imagePreview.style.display =
        "none";

      clearCapturedFrameButton.style.display =
        "none";
    }
  }


  function syncImageSourceControls() {

    const hasCaptured =
      Boolean(capturedImageBytes);

    const hasUploaded =
      Boolean(uploadedImageBytes);


    useCapturedImageRadio.disabled =
      !hasCaptured;

    useUploadedImageRadio.disabled =
      !hasUploaded;


    const explicit =
      getExplicitImageSourceSelection();


    if (
      hasCaptured &&
      !hasUploaded
    ) {

      activeImageSource =
        "captured";

      useCapturedImageRadio.checked =
        true;

    } else if (
      !hasCaptured &&
      hasUploaded
    ) {

      activeImageSource =
        "uploaded";

      useUploadedImageRadio.checked =
        true;

    } else if (
      hasCaptured &&
      hasUploaded
    ) {

      if (explicit) {

        activeImageSource =
          explicit;

      } else if (
        activeImageSource !== "captured" &&
        activeImageSource !== "uploaded"
      ) {

        activeImageSource =
          "captured";

        useCapturedImageRadio.checked =
          true;
      }

    } else {

      activeImageSource = null;

      useCapturedImageRadio.checked =
        false;

      useUploadedImageRadio.checked =
        false;
    }


    refreshPreviewForSelectedImageSource();
  }


  function clearCapturedFrame() {

    if (capturedImageUrl) {

      URL.revokeObjectURL(
        capturedImageUrl
      );
    }


    capturedImageBytes = null;
    capturedImageUrl = null;


    if (uploadedImageBytes) {

      activeImageSource =
        "uploaded";

      useUploadedImageRadio.checked =
        true;

      useCapturedImageRadio.checked =
        false;

    } else {

      activeImageSource = null;

      useCapturedImageRadio.checked =
        false;

      useUploadedImageRadio.checked =
        false;
    }


    refreshPreviewForSelectedImageSource();

    updateButtonStates();

    setStatus(
      "Captured frame cleared."
    );
  }


  function handleImageSourceSelectionChange() {

    if (
      useCapturedImageRadio.checked &&
      capturedImageBytes
    ) {

      activeImageSource =
        "captured";

    } else if (
      useUploadedImageRadio.checked &&
      uploadedImageBytes
    ) {

      activeImageSource =
        "uploaded";
    }


    refreshPreviewForSelectedImageSource();

    updateButtonStates();
  }


  // =========================================================================
  // CAMERA
  // =========================================================================

  async function startCamera() {

    try {

      if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
      ) {

        setStatus(
          "Camera access is unavailable. Open the app through HTTPS or localhost."
        );

        return;
      }


      const devices =
        await navigator.mediaDevices.enumerateDevices();


      if (
        !devices.some(
          device =>
            device.kind === "videoinput"
        )
      ) {

        setStatus(
          "No camera device found."
        );

        return;
      }


      const videoStream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            width: {
              ideal: 1280
            },
            height: {
              ideal: 720
            },
            facingMode: "user"
          }
        });


      if (localStream) {

        videoStream
          .getVideoTracks()
          .forEach(track =>
            localStream.addTrack(track)
          );

      } else {

        localStream =
          videoStream;
      }


      camera.srcObject =
        localStream;

      camera.style.display =
        "block";


      await camera
        .play()
        .catch(() => {});


      setStatus(
        "Camera started."
      );

    } catch (error) {

      console.error(
        "Camera error:",
        error
      );


      if (
        error.name ===
        "NotAllowedError"
      ) {

        setStatus(
          "Camera access denied. Please allow camera permission."
        );

      } else if (
        error.name ===
        "NotFoundError"
      ) {

        setStatus(
          "No camera found on this device."
        );

      } else {

        setStatus(
          `Could not start camera: ${error.message}`
        );
      }
    }


    updateButtonStates();
  }


  // =========================================================================
  // MICROPHONE
  // =========================================================================

  async function startMicrophone() {

    try {

      if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
      ) {

        setStatus(
          "Microphone access is unavailable. Open the app through HTTPS or localhost."
        );

        return;
      }


      const devices =
        await navigator.mediaDevices.enumerateDevices();


      if (
        !devices.some(
          device =>
            device.kind === "audioinput"
        )
      ) {

        setStatus(
          "No microphone found."
        );

        return;
      }


      const audioStream =
        await navigator.mediaDevices.getUserMedia({
          audio: true
        });


      if (localStream) {

        audioStream
          .getAudioTracks()
          .forEach(track =>
            localStream.addTrack(track)
          );

      } else {

        localStream =
          audioStream;
      }


      setStatus(
        "Microphone started."
      );

    } catch (error) {

      console.error(
        "Microphone error:",
        error
      );


      if (
        error.name ===
        "NotAllowedError"
      ) {

        setStatus(
          "Microphone access denied. Please allow microphone permission."
        );

      } else if (
        error.name ===
        "NotFoundError"
      ) {

        setStatus(
          "No microphone found."
        );

      } else {

        setStatus(
          `Could not start microphone: ${error.message}`
        );
      }
    }


    updateButtonStates();
  }


  // =========================================================================
  // START SESSION
  // =========================================================================

  async function startSession() {

    try {

      if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
      ) {

        setStatus(
          "Media devices are unavailable. HTTPS is required for remote access."
        );

        return;
      }


      localStream =
        await navigator.mediaDevices.getUserMedia({
          audio: true,
          video: true
        });


      camera.srcObject =
        localStream;

      camera.style.display =
        "block";


      await camera
        .play()
        .catch(() => {});


      setStatus(
        "Camera and microphone connected."
      );

    } catch (error) {

      console.error(
        "Session error:",
        error
      );


      setStatus(
        `Could not start session: ${error.message}`
      );

      localStream = null;
    }


    updateButtonStates();
  }


  // =========================================================================
  // STOP CAMERA
  // =========================================================================

  function stopCamera() {

    if (!localStream) {

      setStatus(
        "No active session."
      );

      return;
    }


    const tracks =
      localStream
        .getVideoTracks()
        .filter(
          track =>
            track.readyState !== "ended"
        );


    if (!tracks.length) {

      setStatus(
        "Camera already stopped."
      );

      return;
    }


    if (isStreamingVideo) {
      stopVideoStream();
    }


    tracks.forEach(track => {

      track.stop();

      localStream.removeTrack(track);
    });


    camera.srcObject =
      localStream;


    setStatus(
      "Camera stopped."
    );


    if (
      !localStream.getTracks().length
    ) {

      localStream = null;
      camera.srcObject = null;
    }


    updateButtonStates();
  }


  // =========================================================================
  // STOP MICROPHONE
  // =========================================================================

  function stopMicrophone() {

    if (!localStream) {

      setStatus(
        "No active session."
      );

      return;
    }


    const tracks =
      localStream
        .getAudioTracks()
        .filter(
          track =>
            track.readyState !== "ended"
        );


    if (!tracks.length) {

      setStatus(
        "Microphone already stopped."
      );

      return;
    }


    if (isStreamingAudio) {
      stopAudioStream();
    }


    tracks.forEach(track => {

      track.stop();

      localStream.removeTrack(track);
    });


    setStatus(
      "Microphone stopped."
    );


    if (
      !localStream.getTracks().length
    ) {

      localStream = null;
      camera.srcObject = null;
    }


    updateButtonStates();
  }


  // =========================================================================
  // STOP SESSION
  // =========================================================================

  function stopSession() {

    if (!localStream) {

      setStatus(
        "No active session."
      );

      return;
    }


    if (isStreamingVideo) {
      stopVideoStream();
    }


    if (isStreamingAudio) {
      stopAudioStream();
    }


    localStream
      .getTracks()
      .forEach(track =>
        track.stop()
      );


    localStream = null;

    camera.srcObject = null;


    setStatus(
      "Session stopped."
    );


    updateButtonStates();
  }


  // =========================================================================
  // MANUAL IMAGE CAPTURE
  // =========================================================================

  function setCapturedImage(
    blob,
    arrayBuffer
  ) {

    capturedImageBytes =
      arrayBuffer;


    if (capturedImageUrl) {

      URL.revokeObjectURL(
        capturedImageUrl
      );
    }


    capturedImageUrl =
      URL.createObjectURL(blob);


    activeImageSource =
      "captured";


    useCapturedImageRadio.checked =
      true;


    syncImageSourceControls();
  }


  async function captureFace() {

    if (
      !localStream ||
      !localStream.getVideoTracks().length
    ) {

      setStatus(
        "Start the session before capturing a frame."
      );

      return;
    }


    const blob =
      await grabFrameBlob();


    if (!blob) {

      setStatus(
        "Camera feed is not ready yet."
      );

      return;
    }


    setCapturedImage(
      blob,
      await blob.arrayBuffer()
    );


    setStatus(
      "Frame extracted."
    );


    updateButtonStates();
  }


  // =========================================================================
  // UPLOAD IMAGE
  // =========================================================================

  async function updateUploadedImagePreview() {

    const file =
      imageUploadInput.files[0];


    if (!file) {

      uploadedImageBytes = null;


      if (uploadedImageUrl) {

        URL.revokeObjectURL(
          uploadedImageUrl
        );

        uploadedImageUrl = null;
      }


      syncImageSourceControls();

      updateButtonStates();

      return;
    }


    if (uploadedImageUrl) {

      URL.revokeObjectURL(
        uploadedImageUrl
      );
    }


    uploadedImageUrl =
      URL.createObjectURL(file);


    uploadedImageBytes =
      await file.arrayBuffer();


    if (
      !capturedImageBytes &&
      !activeImageSource
    ) {

      activeImageSource =
        "uploaded";

      useUploadedImageRadio.checked =
        true;
    }


    if (
      activeImageSource ===
      "uploaded"
    ) {

      imagePreview.src =
        uploadedImageUrl;

      imageSourceLabel.textContent =
        "Uploaded image";

      imagePreview.style.display =
        "block";
    }


    syncImageSourceControls();

    setStatus(
      "Image selected."
    );

    updateButtonStates();
  }


  async function analyzeUploadedImage() {

    const file =
      imageUploadInput.files[0];


    if (!file) {

      setStatus(
        "Select an image before analyzing."
      );

      return;
    }


    try {

      uploadedImageBytes =
        uploadedImageBytes ||
        await file.arrayBuffer();


      if (!uploadedImageUrl) {

        uploadedImageUrl =
          URL.createObjectURL(file);
      }


      syncImageSourceControls();


      const response =
        await fetch(
          "/api/image",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/octet-stream"
            },

            body:
              uploadedImageBytes
          }
        );


      if (!response.ok) {

        setStatus(
          `Image analysis failed: ${await response.text()}`
        );

        return;
      }


      const data =
        await response.json();


      const metadata =
        data.metadata || {};


      emotionMetadata.textContent =
        JSON.stringify(
          {
            source:
              metadata.source,

            text_emotion:
              metadata.text_emotion,

            fused_emotion:
              metadata.fused_emotion
          },
          null,
          2
        );


      setStatus(
        "Uploaded image analyzed."
      );

    } catch (error) {

      setStatus(
        `Image analysis failed: ${error.message}`
      );
    }


    updateButtonStates();
  }


  // =========================================================================
  // SEND CAPTURED FRAME
  // =========================================================================

  async function sendFrame() {

    if (!capturedImageBytes) {

      setStatus(
        "Extract a frame first."
      );

      return;
    }


    try {

      const response =
        await fetch(
          "/api/image",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/octet-stream"
            },

            body:
              capturedImageBytes
          }
        );


      if (!response.ok) {

        setStatus(
          `Frame request failed: ${await response.text()}`
        );

        return;
      }


      const data =
        await response.json();


      const metadata =
        data.metadata || {};


      emotionMetadata.textContent =
        JSON.stringify(
          {
            source:
              metadata.source,

            text_emotion:
              metadata.text_emotion,

            fused_emotion:
              metadata.fused_emotion
          },
          null,
          2
        );


      setStatus(
        "Captured frame sent and analyzed."
      );

    } catch (error) {

      setStatus(
        `Frame request failed: ${error.message}`
      );
    }


    updateButtonStates();
  }


  // =========================================================================
  // MANUAL AUDIO RECORDING
  // =========================================================================

  function createMediaRecorder() {

    const tracks =
      localStream
        ?.getAudioTracks()
        .filter(
          track =>
            track.readyState !== "ended"
        ) || [];


    if (!tracks.length) {
      return null;
    }


    const audioStream =
      new MediaStream([
        tracks[0]
      ]);


    const options = {};


    if (
      MediaRecorder.isTypeSupported(
        "audio/webm;codecs=opus"
      )
    ) {

      options.mimeType =
        "audio/webm;codecs=opus";

    } else if (
      MediaRecorder.isTypeSupported(
        "audio/webm"
      )
    ) {

      options.mimeType =
        "audio/webm";
    }


    try {

      const recorder =
        new MediaRecorder(
          audioStream,
          options
        );


      recorder.ondataavailable =
        event => {

          if (
            event.data &&
            event.data.size > 0
          ) {

            audioChunks.push(
              event.data
            );

            updateButtonStates();
          }
        };


      recorder.onstop = () => {

        setStatus(
          "Recording stopped. Ready to send."
        );

        updateButtonStates();
      };


      recorder.onerror = event => {

        setStatus(
          `Recording error: ${event.error?.message || "Unknown"}`
        );
      };


      return recorder;

    } catch (error) {

      setStatus(
        `Could not create recorder: ${error.message}`
      );

      return null;
    }
  }


  async function startVoiceRecording() {

    if (
      !localStream ||
      !localStream.getAudioTracks().length
    ) {

      setStatus(
        "Start the session before recording."
      );

      return;
    }


    audioChunks = [];


    mediaRecorder =
      createMediaRecorder();


    if (!mediaRecorder) {
      return;
    }


    try {

      mediaRecorder.start();

      setStatus(
        "Recording voice..."
      );

    } catch (error) {

      setStatus(
        `Could not start recording: ${error.message}`
      );

      mediaRecorder = null;
    }


    updateButtonStates();
  }


  function stopVoiceRecording() {

    if (
      !mediaRecorder ||
      mediaRecorder.state !== "recording"
    ) {

      setStatus(
        "No active recording."
      );

      return;
    }


    try {

      mediaRecorder.stop();

    } catch (error) {

      setStatus(
        `Stop failed: ${error.message}`
      );
    }


    updateButtonStates();
  }


  async function sendVoiceInput() {

    if (!audioChunks.length) {

      setStatus(
        "Record audio before sending."
      );

      return;
    }


    try {

      const blob =
        new Blob(
          audioChunks,
          {
            type:
              mediaRecorder?.mimeType ||
              "audio/webm"
          }
        );


      const response =
        await fetch(
          "/api/audio",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/octet-stream"
            },

            body:
              await blob.arrayBuffer()
          }
        );


      if (!response.ok) {

        setStatus(
          `Voice request failed: ${await response.text()}`
        );

        return;
      }


      const data =
        await response.json();


      voiceTranscript.textContent =
        data.transcript ||
        "No transcript returned.";


      responseText.textContent =
        data.response || "";


      const metadata =
        data.metadata || {};


      emotionMetadata.textContent =
        JSON.stringify(
          {
            source:
              metadata.source,

            text_emotion:
              metadata.text_emotion,

            fused_emotion:
              metadata.fused_emotion
          },
          null,
          2
        );


      setStatus(
        "Voice input processed."
      );


      audioChunks = [];

    } catch (error) {

      setStatus(
        `Voice request failed: ${error.message}`
      );
    }


    updateButtonStates();
  }


  // =========================================================================
  // MULTIMODAL HTTP REQUEST
  // =========================================================================

  async function sendMultimodalInput() {

    const text =
      messageInput.value.trim();


    if (!text) {

      setStatus(
        "Enter a message first."
      );

      return;
    }


    const formData =
      new FormData();


    formData.append(
      "text",
      text
    );


    let imageBytes =
      activeImageSource === "uploaded"
        ? uploadedImageBytes
        : activeImageSource === "captured"
          ? capturedImageBytes
          : uploadedImageBytes ||
            capturedImageBytes;


    if (imageBytes) {

      formData.append(
        "image",
        new Blob(
          [imageBytes],
          {
            type: "image/jpeg"
          }
        ),
        "image.jpg"
      );
    }


    if (audioChunks.length) {

      const blob =
        new Blob(
          audioChunks,
          {
            type:
              mediaRecorder?.mimeType ||
              "audio/webm"
          }
        );


      formData.append(
        "audio",
        blob,
        "audio.webm"
      );
    }


    try {

      const response =
        await fetch(
          "/api/multimodal",
          {
            method: "POST",
            body: formData
          }
        );


      if (!response.ok) {

        setStatus(
          `Multimodal request failed: ${await response.text()}`
        );

        return;
      }


      const data =
        await response.json();


      voiceTranscript.textContent =
        data.transcript ||
        "No transcript returned.";


      responseText.textContent =
        data.response || "";


      const metadata =
        data.metadata || {};


      emotionMetadata.textContent =
        JSON.stringify(
          {
            source:
              metadata.source,

            face_emotion:
              metadata.face_emotion,

            speech_sentiment:
              metadata.speech_sentiment,

            text_emotion:
              metadata.text_emotion,

            fused_emotion:
              metadata.fused_emotion
          },
          null,
          2
        );


      setStatus(
        "Multimodal input processed."
      );


      audioChunks = [];

    } catch (error) {

      setStatus(
        `Multimodal request failed: ${error.message}`
      );
    }


    updateButtonStates();
  }


  // =========================================================================
  // TEXT ONLY
  // =========================================================================

  async function sendTextMessage() {

    const text =
      messageInput.value.trim();


    if (!text) {

      setStatus(
        "Enter a message first."
      );

      return;
    }


    sendTextButton.disabled =
      true;


    try {

      const response =
        await fetch(
          "/api/text",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body:
              JSON.stringify({
                text
              })
          }
        );


      if (!response.ok) {

        setStatus(
          `Text request failed: ${await response.text()}`
        );

        return;
      }


      const data =
        await response.json();


      responseText.textContent =
        data.response || "";


      const metadata =
        data.metadata || {};


      emotionMetadata.textContent =
        JSON.stringify(
          {
            source:
              metadata.source,

            text_emotion:
              metadata.text_emotion,

            fused_emotion:
              metadata.fused_emotion
          },
          null,
          2
        );


      setStatus(
        "Text sent and response generated."
      );

    } catch (error) {

      setStatus(
        `Text request failed: ${error.message}`
      );

    } finally {

      sendTextButton.disabled =
        false;
    }
  }


  // =========================================================================
  // MODE TOGGLE (Manual vs Live Stream)
  // =========================================================================
  //
  // Purely a visibility switch over the button groups tagged data-mode
  // in index.html. It does NOT stop camera/mic acquisition or an
  // in-progress live stream when you switch away from it — Start/Stop
  // Camera and Microphone stay common to both modes on purpose, and a
  // stream you started stays running in the background until you switch
  // back to Live Stream and stop it explicitly.

  function setMode(mode) {
    if (!controlRail) return;

    const isManual = mode === "manual";

    controlRail.classList.toggle("mode-manual", isManual);
    controlRail.classList.toggle("mode-live", !isManual);

    modeManualButton?.classList.toggle("is-active", isManual);
    modeManualButton?.setAttribute("aria-selected", String(isManual));

    modeLiveButton?.classList.toggle("is-active", !isManual);
    modeLiveButton?.setAttribute("aria-selected", String(!isManual));
  }

  modeManualButton?.addEventListener(
    "click",
    () => setMode("manual")
  );

  modeLiveButton?.addEventListener(
    "click",
    () => setMode("live")
  );


  // =========================================================================
  // EVENT LISTENERS
  // =========================================================================

  startButton?.addEventListener(
    "click",
    startSession
  );


  startCameraButton?.addEventListener(
    "click",
    startCamera
  );


  startMicrophoneButton?.addEventListener(
    "click",
    startMicrophone
  );


  stopCameraButton?.addEventListener(
    "click",
    stopCamera
  );


  stopMicrophoneButton?.addEventListener(
    "click",
    stopMicrophone
  );


  stopSessionButton?.addEventListener(
    "click",
    stopSession
  );


  captureImageButton?.addEventListener(
    "click",
    captureFace
  );


  uploadImageButton?.addEventListener(
    "click",
    analyzeUploadedImage
  );


  sendFrameButton?.addEventListener(
    "click",
    sendFrame
  );


  clearCapturedFrameButton?.addEventListener(
    "click",
    clearCapturedFrame
  );


  sendTextButton?.addEventListener(
    "click",
    sendTextMessage
  );


  sendMultimodalButton?.addEventListener(
    "click",
    sendMultimodalInput
  );


  startRecordingButton?.addEventListener(
    "click",
    startVoiceRecording
  );


  stopRecordingButton?.addEventListener(
    "click",
    stopVoiceRecording
  );


  sendVoiceButton?.addEventListener(
    "click",
    sendVoiceInput
  );


  imageUploadInput?.addEventListener(
    "change",
    updateUploadedImagePreview
  );


  useCapturedImageRadio?.addEventListener(
    "change",
    handleImageSourceSelectionChange
  );


  useUploadedImageRadio?.addEventListener(
    "change",
    handleImageSourceSelectionChange
  );


  messageInput?.addEventListener(
    "input",
    updateButtonStates
  );


  streamVideoBtn?.addEventListener(
    "click",
    startVideoStream
  );


  streamAudioBtn?.addEventListener(
    "click",
    startAudioStream
  );


  // =========================================================================
  // INITIAL STATE
  // =========================================================================

  updateButtonStates();

});