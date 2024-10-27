import React, { useEffect, useRef } from 'react';
import axios from 'axios';

function Webcamcapture() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    const startVideo = () => {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          videoRef.current.srcObject = stream;
        })
        .catch((err) => {
          console.error("Error accessing the camera: ", err);
          alert("Could not access the camera. Please allow camera permissions.");
        });
    };

    const captureImage = async () => {
        const canvas = canvasRef.current;
        const video = videoRef.current;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Convert the canvas image to a blob and send it to the server
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'capture.jpg');

            try {
                const response = await axios.post('http://localhost:5000//detect', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                });
                console.log('Detection results:', response.data);
                // Handle the detection results (draw boxes, etc.)
            } catch (error) {
                console.error('Error during detection:', error);
            }
        }, 'image/jpeg');
    };
    useEffect(()=>{
        startVideo();
    },[])

    return (
        <div>
            <h1>Face and Hand Detection</h1>
            <video ref={videoRef} width="640" height="480"
            className="w-80 h-60 border-2 border-green-500"
            autoPlay/>
            <button onClick={captureImage}>Capture and Detect</button>
            <canvas ref={canvasRef} style={{display:"none"}} />
        </div>
    );
}

export default Webcamcapture;
