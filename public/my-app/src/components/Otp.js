// import React, { useState ,useRef,useEffect} from 'react';
// import { useNavigate } from "react-router-dom";
// import { TextField, Button, Box, Typography, Grid, Paper } from '@mui/material';
// import './OtpPage.css';
// import axios from 'axios';

// const Otp = () => {
//   const [otp, setOtp] = useState(new Array(6).fill(""));
//   const [error, setError] = useState(false);
//   const [success, setSuccess] = useState(''); // State for success messages
//   const navigate = useNavigate();
//   const handleChange = (element, index) => {
//     if (isNaN(element.value)) return; // Allow only numbers
//     setOtp([...otp.map((d, idx) => (idx === index ? element.value : d))]);

//     // Move to the next input field automatically
//     if (element.nextSibling && element.value) {
//       element.nextSibling.focus();
//     }
//   };

//   const handleSubmit1 = async(e) => {
//     e.preventDefault();
//     const otpValue = otp.join("");
//     if (otpValue.length === 6) {
//       setError(false);
//       console.log("Entered OTP:", otpValue);
//       // Here you can add the API call to verify the OTP
//       try {
//         const response = await axios.post('https://api.example.com/verify-otp', {
//           otpValue, // Send the OTP in the request body
//         });
  
//         // Assuming a successful response
//         if (response.data.success) {
//           setSuccess('OTP verified successfully!');
//           startRecording();
//           navigate("/home");
//           setError(''); // Clear any previous errors
//         } else {
//           setError('Invalid OTP. Please try again.');
//           setSuccess('');
//         }
//       } catch (error) {
//         stopCapture();
//         stopRecording();
//         console.error('Error verifying OTP:', error);
//         setError('An error occurred while verifying OTP. Please try again.');
//         setSuccess('');
//       }
//     } else {
//       setError(true);
//     }
//   };


//   const [isRecording, setIsRecording] = useState(false);
//   const [videoURL, setVideoURL] = useState(null);
//   const mediaRecorderRef = useRef(null);
//   const recordedChunks = useRef([]);

//   const startRecording = async () => {
//     try {
//       // Capture the screen
//       const stream = await navigator.mediaDevices.getDisplayMedia({
//         video: { mediaSource: 'screen' },
//       });

//       // Initialize MediaRecorder
//       mediaRecorderRef.current = new MediaRecorder(stream, {
//         mimeType: 'video/webm',
//       });

//       // Collect the recorded chunks of data
//       mediaRecorderRef.current.ondataavailable = (event) => {
//         if (event.data.size > 0) {
//           recordedChunks.current.push(event.data);
//         }
//       };

//       // Handle the stop event
//       mediaRecorderRef.current.onstop = () => {
//         // Create a video URL from the recorded chunks
//         const blob = new Blob(recordedChunks.current, { type: 'video/webm' });
//         const videoURL = URL.createObjectURL(blob);
//         setVideoURL(videoURL);
//         recordedChunks.current = []; // Clear the recorded chunks
//       };

//       // Start the recording
//       mediaRecorderRef.current.start();
//       setIsRecording(true);
//     } catch (err) {
//       console.error('Error accessing display media', err);
//     }
//   };

//   const stopRecording = () => {
//     // Stop the recording
//     mediaRecorderRef.current.stop();
//     mediaRecorderRef.current = null; 
//     setIsRecording(false);
//   };

//   const containerRef = useRef(null);
//   //fullscreen code
//   useEffect(() => {
//     // Request full-screen mode when the component mounts
//     if (containerRef.current) {
//       containerRef.current.requestFullscreen();
//     }

//     // Disable right-click context menu
//     const handleContextMenu = (e) => {
//       e.preventDefault();
//     };

//     // Disable certain key combinations
//     const handleKeyDown = (e) => {
//       // Disable F11, Ctrl+T, Ctrl+N, Ctrl+W, and Alt+F4
//       if (
//         e.key === 'F11' ||
//         (e.ctrlKey && (e.key === 't' || e.key === 'n' || e.key === 'w')) ||
//         (e.altKey && e.key === 'F4')
//       ) {
//         e.preventDefault();
//       }
//     };

//     window.addEventListener('contextmenu', handleContextMenu);
//     window.addEventListener('keydown', handleKeyDown);

//     return () => {
//       // Clean up the event listeners when the component unmounts
//       window.removeEventListener('contextmenu', handleContextMenu);
//       window.removeEventListener('keydown', handleKeyDown);
//     };
//   }, []);


//   //cameralive
//   const videoRef = useRef(null);
//   const streamRef = useRef(null);
//   useEffect(() => {
//     navigator.mediaDevices.getUserMedia({ video: true })
//       .then((stream) => {
//         videoRef.current.srcObject = stream;
//       })
//       .catch((err) => {
//         console.error("Error accessing the camera: ", err);
//         alert("Could not access the camera. Please allow camera permissions.");
//       });
//   }, []);

//   const stopCapture = () => {
//     if (streamRef.current) {
//       // Stop all tracks in the media stream
//       streamRef.current.getTracks().forEach((track) => track.stop());
//       videoRef.current.srcObject = null; // Clear the video element's source
//       streamRef.current = null; // Clear the stream reference
//     }
//   };

//   return (
//     <>
    
//     <Grid container justifyContent="center" alignItems="center" style={{ minHeight: "100vh" }}>
//         <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '10px' ,marginRight:"40px"}}>
//             <video ref={videoRef} width="320" height="240" autoPlay style={{ border: "2px solid blue" }}></video>
//         </div>
//       <Paper elevation={3} className="otp-paper">
//         <Typography variant="h4" align="center" gutterBottom>
//           Enter OTP
//         </Typography>
//         <Typography variant="body1" align="center" color="textSecondary">
//           Please enter the 6-digit code sent to your mobile number
//         </Typography>
//         <form onSubmit={handleSubmit1} style={{ marginTop: "20px" }}>
//           <Box display="flex" justifyContent="center" mb={2}>
//             {otp.map((data, index) => (
//               <TextField
//                 key={index}
//                 type="text"
//                 inputProps={{ maxLength: 1, style: { textAlign: 'center', fontSize: '24px' } }}
//                 value={data}
//                 onChange={(e) => handleChange(e.target, index)}
//                 onFocus={(e) => e.target.select()}
//                 className="otp-input"
//               />
//             ))}
//           </Box>
//           {error && <Typography color="error" align="center">Please enter all 6 digits</Typography>}
//           <Box textAlign="center">
//             <Button type="submit" variant="contained" color="primary" className="submit-btn">
//               Verify OTP
//             </Button>
//           </Box>
//         </form>
//       </Paper>
//     </Grid>
//     </>
//   );
// };

// export default Otp;
import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from "react-router-dom";
import axios from 'axios';

const Otp = () => {
  const [otp, setOtp] = useState(new Array(6).fill(""));
  const [error, setError] = useState(false);
  const [success, setSuccess] = useState(''); // State for success messages
  const navigate = useNavigate();

  const handleChange = (element, index) => {
    if (isNaN(element.value)) return; // Allow only numbers
    setOtp([...otp.map((d, idx) => (idx === index ? element.value : d))]);

    // Move to the next input field automatically
    if (element.nextSibling && element.value) {
      element.nextSibling.focus();
    }
  };

  const handleSubmit1 = async (e) => {
    e.preventDefault();
    const otpValue = otp.join("");
    if (otpValue.length === 6) {
      setError(false);
      console.log("Entered OTP:", otpValue);

      try {
        const response = await axios.post('https://api.example.com/verify-otp', {
          otpValue, // Send the OTP in the request body
        });
        if (response.data.success) {
          setSuccess('OTP verified successfully!');
          alert(success);
          navigate("/camera");
          setError(''); // Clear any previous errors
        } else {
          setError('Invalid OTP. Please try again.');
          setSuccess('');
        }
      } catch (error) {
        
        localStorage.removeItem('token');
        navigate('/signup');
        alert('An error occurred while verifying OTP. Please try with different credentials.');
        window.location.reload();
        console.error('Error verifying OTP:', error);
        setError('An error occurred while verifying OTP. Please try again.');
        setSuccess('');
      }
    } else {
      setError(true);
    }
  };

  const videoRef = useRef(null);

  useEffect(() => {
    if(localStorage.getItem("token")){
      navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
      })
      .catch((err) => {
        console.error("Error accessing the camera: ", err);
        alert("Could not access the camera. Please allow camera permissions.");
      });
    }
  }, []);


  return (
    <div className="flex items-center justify-center h-full py-20">
      <div className="flex flex-wrap justify-center gap-2 md:gap-4 lg:gap-6 mr-12">
        <video
          ref={videoRef}
          className="w-80 h-60 border-2 border-green-500"
          autoPlay
        ></video>
      </div>

      <div className="bg-white shadow-md p-6 rounded-lg max-w-md w-full">
        <h4 className="text-2xl font-bold text-center mb-4">Enter OTP</h4>
        <p className="text-center text-gray-600 mb-4">
          Please enter the 6-digit code sent to your mobile number
        </p>
        <form onSubmit={handleSubmit1} className="mt-5">
          <div className="flex justify-center mb-4 space-x-2">
            {otp.map((data, index) => (
              <input
                key={index}
                type="text"
                maxLength={1}
                value={data}
                onChange={(e) => handleChange(e.target, index)}
                onFocus={(e) => e.target.select()}
                className="w-12 h-12 text-center text-xl border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            ))}
          </div>
          {error && <p className="text-red-500 text-center">Please enter all 6 digits</p>}
          <div className="text-center">
            <button type="submit" className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600">
              Verify OTP
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Otp;
