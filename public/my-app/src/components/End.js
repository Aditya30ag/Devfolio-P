// import React from "react";

// export default function End(props) {
//   /*const logout = () => {
//     localStorage.removeItem("token");
//     navigate("/login");
//     props.showalert();
//   };*/
//   return (
//     <div
//       className="container"
//       style={{
//         display: "flex",
//         justifyContent: "center",
//         alignItems: "center",
//         flexDirection: "column",
//         marginTop:"20%",
//         rowGap:"20px"
//       }}
//     >
//       <div>
//         <h2>Thank you</h2>
//       </div>
//       <div>
//         <h3>Your Vote is added</h3>
//       </div>
//     </div>
//   );
// }
import React from "react";
import { useNavigate } from "react-router-dom";

export default function End() {
  const navigate=useNavigate();
  const onAction=()=>{
    localStorage.removeItem('token');
    navigate('/');
  }
  return (
    <div className="flex items-center justify-center my-40">
      <div className="text-center bg-white p-10 rounded-lg shadow-lg max-w-md">
        <div className="mb-5">
          <h2 className="text-2xl font-bold text-gray-800">Thank You!</h2>
        </div>
        <div className="mb-5">
          <h3 className="text-lg text-gray-600">Your vote has been successfully added.</h3>
        </div>
        <button
          className="mt-6 px-6 py-3 bg-blue-600 text-white font-medium rounded-lg shadow hover:bg-blue-700 transition duration-300"
          onClick={onAction} // optional, if you want to provide any action
        >
          Go Back to Homepage
        </button>
      </div>
      </div>
  );
}

