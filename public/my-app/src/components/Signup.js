// import React, { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import "./logout.css";
// import axios from 'axios';

// export default function Signup(props) {
//   const [credentials, setcredential] = useState({
//     voterid:"",
//     aadharNumber: "",
//   });
//   const password = document.querySelector("#exampleInputEmail1");
//   const navigate = useNavigate();

//   //Otp genrator
//   const forotp = async (e) => {
//     e.preventDefault();

//     try {
//       // Make an API call to the backend to send OTP to the provided email
//       const response = await axios.post('http://localhost:5000/send-otp', credentials.email);
//       localStorage.setItem("sentotp","otp");
//     } catch (err) {
//       console.log("error");
//     }
//   };

//   const handleonClick = async (e) => {
//     e.preventDefault();
//     const url = "http://localhost:5000/api/auth/createuser";
//     const response = await fetch(url, {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({
//         voterid:credentials.voterid,
//         aadharNumber: credentials.aadharNumber,
//       }),
//     });
//     const json = await response.json();
//     if (json.success === true) {
//       localStorage.setItem("token", json.token);
//       if (localStorage.getItem('token')) {
//         forotp(e);
//         navigate("/otp");
//       }
//       props.handleonClick2();
//     } else {
//       props.showalert();
//       password.style.border = "1px solid red";
//     }
//   };
//   const onchange = async (e) => {
//     await setcredential({ ...credentials, [e.target.name]: e.target.value });
//   };
//   return (
//     <>
//       <div className="container1" style={{width:"100%",display:"flex",justifyContent:"center",alignItems:"center",zIndex:"10",marginTop:"50px"}}>
//       <div
//         className="container"
//         style={{
//           marginTop: "50px",
//           maxWidth: "500px",
//           minHeight: "300px",
//           border: "2px solid #EAEAEA",
//           boxSizing: "border-box",
//           paddingTop: "20px",
//           borderRadius: "12px",
//           boxShadow: "0px 10px 25px #000",
//           backgroundColor:"whitesmoke",//#E4F1E8
//           opacity:"0.9",
//           fontWeight:"bolder"
//         }}
//       >
//           <form onSubmit={handleonClick}>
//             <div className="mb-3 my-2">
//               <label htmlFor="exampleInputvoterid" className="form-label">
//                 Voter Id
//               </label>
//               <input
//                 type="password"
//                 className="form-control"
//                 id="exampleInputvoterid"
//                 name="voterid"
//                 value={credentials.voterid}
//                 onChange={onchange}
//               />
//             </div>
//             <div className="mb-3 my-2">
//               <label htmlFor="exampleInputaadharNumber1" className="form-label">
//                 AadharNumber
//               </label>
//               <input
//                 type="password"
//                 className="form-control"
//                 id="exampleInputaadharNumber1"
//                 name="aadharNumber"
//                 value={credentials.aadharNumber}
//                 onChange={onchange}
//               />
//             </div>
//             <button
//               type="submit"
//               className="same-btn1"
//               onClick={handleonClick}
//               style={{marginTop:"40px"}}
//             >
//               Done
//             </button>
//           </form>
//         </div>
//       </div>
//     </>
//   );
// }
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function Signup(props) {
  const [credentials, setCredential] = useState({
    voterid: "",
    aadharNumber: "",
  });
  const navigate = useNavigate();

  // Otp generator function
  const forotp = async (e) => {
    e.preventDefault();
    try {
      // API call to backend to send OTP
      const response = await axios.post('http://localhost:5000/send-otp', { email: credentials.email });
      localStorage.setItem("sentotp", "otp");
    } catch (err) {
      console.log("Error sending OTP");
    }
  };

  const [content, setcontent] = useState("");
  // Handle form submission
  const handleOnClick = async (e) => {
    e.preventDefault();
    const url = "http://localhost:5000/api/auth/login";
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        voterid:credentials.voterid,
        aadharNumber: credentials.aadharNumber,
      }),
    });
    const json = await response.json();
    console.log(json)
    if (json.success === true) {
      localStorage.setItem("token", json.token);
      navigate("/otp");
      props.showalert();
    } else {
      props.showalert();
      if(json.success1===false){
        setcontent(json.error);
      }
      setcontent(json.error);
    }
  };

  // Handle input change
  const handleChange = (e) => {
    setCredential({ ...credentials, [e.target.name]: e.target.value });
  };

  return (
    <div className="flex items-center justify-center h-full py-20">
      <div className="bg-white rounded-lg shadow-xl p-8 max-w-lg w-full">
        <h2 className="text-3xl font-bold text-center text-gray-800 mb-6">
        Be Heard, Be Counted, Vote
        </h2>
        <form onSubmit={handleOnClick} className="space-y-6">
          <div>
            <label htmlFor="voterid" className="block text-gray-700 font-semibold mb-2">
              Voter ID
            </label>
            <input
              type="text"
              id="voterid"
              name="voterid"
              value={credentials.voterid}
              onChange={handleChange}
              className="w-full p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
          <div>
            <label htmlFor="aadharNumber" className="block text-gray-700 font-semibold mb-2">
              Aadhar Number
            </label>
            <input
              type="text"
              id="aadharNumber"
              name="aadharNumber"
              value={credentials.aadharNumber}
              onChange={handleChange}
              className="w-full p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
          <button
            type="submit"
            className="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-300"
          >
            Done
          </button>
        </form>
      </div>
    </div>
  );
}
