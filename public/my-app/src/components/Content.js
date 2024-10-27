// import React, { useEffect,useContext,useState,useRef} from "react";
// import { useNavigate } from "react-router-dom";
// import Noteitem from "./Noteitem";
// import Notecontext from "../context/notes/NotesContext";

// export default function Content(props) {

//   const a = useContext(Notecontext);
//   const navigate = useNavigate();

//   useEffect(() => {
//     if (localStorage.getItem("token")) {
//       a.getNotes();
//     } else {
//       navigate("/");
//     }
//     // eslint-disable-next-line
//   }, []);


//   const [disable,setdisable]=useState(false);
//   const [newvalue,setnewvalue]=useState();
//   const handleonclick=async(note)=>{
//     //e.preventDefault()
//     //console.log(e.note)
//     console.log(note);
//     setnewvalue(note);
//     await a.getuser()
//     setdisable(true)
//   }
//   const handleonclick2=async(e)=>{
//     e.preventDefault()
//     if(handleonclick){
//       const c=a.user._id
//       console.log(c);
//       await newvalue.voteCount++;
//       await a.updatecandidate(newvalue._id,newvalue.name,newvalue.party,newvalue.voteCount);
//       await a.updateuser(c,a.user.name,a.user.email,a.user.aadharNumber,a.user.isvoted);
//       setdisable(false);
//       localStorage.removeItem("token");
//       stopCapture();
//       navigate("/end");
//       props.handleonClick2();
//     }else{

//     }
//   }

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
//     <div className="container my-4" style={{display:"flex",justifyContent:"center",alignItems:"center"}}>
    
//       <div className="container" style={{display:"flex",justifyContent:"center",alignItems:"center",flexDirection:"column",boxSizing: "border-box"}}>
//         { 
//           a.notes.map((note) => {
//             return (
//               <div key={note._id}>
//               <Noteitem
//                 key={note._id}
//                 note={note}
//                 name={note.name}
//                 party={note.party}
//                 handleonClick2={props.handleonClick2}
//                 handleonclick={handleonclick}
//                 disable={disable}
//               />
//               </div>
//             );
//           })
//         }
//       </div>
//       <div ><button className='btn btn-primary' onClick={handleonclick2} type="submit" style={{position:"relative",bottom:"20px",marginLeft:"10px"}}>Submit</button></div>
//     </div>
//     <div style={{position:"absolute",top:"200px"}}>
//         <video ref={videoRef} width="320" height="240" autoPlay style={{ border: "2px solid blue" }}></video>
//     </div>
//     </>
//   );
// }
import React, { useEffect, useContext, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import Noteitem from "./Noteitem";
import Notecontext from "../context/notes/NotesContext";

export default function Content(props) {
  const a = useContext(Notecontext);
  const navigate = useNavigate();

  useEffect(() => {
    if (localStorage.getItem("token")) {
      a.getNotes();
    } else {
      navigate("/");
    }
  }, [a, navigate]);

  const [disable, setDisable] = useState(false);
  const [newvalue, setNewvalue] = useState();

  const handleonclick = async (note) => {
    setNewvalue(note);
    await a.getuser();
    setDisable(true);
  };

  const handleonclick2 = async (e) => {
    e.preventDefault();
    if (handleonclick) {
      const c = a.user._id;
      await newvalue.voteCount++;
      await a.updatecandidate(newvalue._id, newvalue.name, newvalue.party, newvalue.voteCount);
      await a.updateuser(c, a.user.name, a.user.email, a.user.aadharNumber, a.user.isvoted);
      setDisable(false);
      localStorage.removeItem("token");
      navigate("/end");
      window.location.reload();
      props.handleonClick2();
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
    <>
      <div className="container my-4 flex justify-center items-center">
        <div className="flex flex-wrap justify-center gap-2 md:gap-4 lg:gap-6 mr-15">
          <video
            ref={videoRef}
            className="w-80 h-60 border-2 border-green-500"
            autoPlay
          ></video>
        </div>
        <div className="container flex flex-col justify-center items-center box-border">
          {a.notes.map((note) => {
            return (
              <div key={note._id}>
                <Noteitem
                  key={note._id}
                  note={note}
                  name={note.name}
                  party={note.party}
                  handleonClick2={props.handleonClick2}
                  handleonclick={handleonclick}
                  disable={disable}
                />
              </div>
            );
          })}
        </div>
        <div>
          <button
            className="btn btn-primary relative bottom-5 ml-3"
            onClick={handleonclick2}
            type="submit"
          >
            Submit
          </button>
        </div>
      </div>
      <div className="absolute top-52">
        <video ref={videoRef} width="320" height="240" autoPlay className="border-2 border-blue-500"></video>
      </div>
    </>
  );
}
