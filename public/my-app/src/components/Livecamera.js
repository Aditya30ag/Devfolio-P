import React, { useState, useRef, useEffect } from 'react';

const Livecamera = () => {
  const [recording, setRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const mediaRecorderRef = useRef(null);

  const handleStartRecording = async () => {
    try {
      // Request screen capture
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { mediaSource: 'screen' },
      });

      // Initialize MediaRecorder and store it in the ref
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm; codecs=vp9', // Use vp9 for better compression, or vp8 for compatibility
      });
      mediaRecorderRef.current = mediaRecorder;

      // Handle data when available
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setRecordedChunks((prev) => [...prev, event.data]);
        }
      };

      // Start recording
      mediaRecorder.start();
      videoRef.current.srcObject = stream;
      setRecording(true);
    } catch (error) {
      console.error("Error starting screen recording:", error);
    }
  };

  const handleStopRecording = () => {
    // Stop the media recorder
    if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
    } else {
        console.error('mediaRecorder is not initialized');
    }
  };

  

  const handleDownload = () => {
    // Create a blob from recorded chunks
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);

    // Create an anchor element for downloading
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'screen_recording.webm';

    // Trigger download and clean up
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
  };
  
  useEffect(() => {
    handleStartRecording();
    if(!localStorage.getItem('token')){
      handleStopRecording();
      if(recordedChunks.length > 0){
        handleDownload();
      }
    }
  },[])
  const videoRef = useRef(null);
  
  return (
    <div className="absolute top-[450px] left-[280px]">
    <video
        ref={videoRef}
        className="w-80 h-50 border-2 border-green-500"
        autoPlay
    ></video>
</div>


  );
};

export default Livecamera;
