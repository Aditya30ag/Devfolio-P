// import React, { useEffect } from "react";
// import { useNavigate } from "react-router-dom";

// export default function Main() {
//   const navigate = useNavigate();

//   useEffect(() => {
//     if (localStorage.getItem("token")) {
//       navigate("/home");
//     } else {
//       navigate("/");
//     }
//   }, [navigate]);

//   const handleonclickdir = () => {
//     navigate('/signup');
//   };

//   return (
//     <div
//       id="photo"
//       className="bg-gradient-to-b from-[#E2EAF4] to-white flex items-center"
//     >
//       <div className="container mx-auto">
//         <div className="text-black text-7xl font-sans mt-5">
//           Welcome to
//         </div>
//         <div className="mx-3">
//           Voting app
//         </div>
//         <div
//           className="mx-3 my-3 cursor-pointer"
//           role="button"
//         >
//           <div
//             className="text-black text-sm no-underline p-2 border-2 border-white rounded-md w-[120px] bg-gray-400"
//             onClick={handleonclickdir}
//           >
//             GET Started <i className="fa-solid fa-arrow-right"></i>
//           </div>
//         </div>
//       </div>
//       <div className="image"></div>
//     </div>
//   );
// }
import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";


export default function Main() {
  const navigate = useNavigate();

  useEffect(() => {
    if (localStorage.getItem("token")) {
      navigate("/home");
    } else {
      navigate("/");
    }
  }, [navigate]);

  const handleonclickdir = () => {
    navigate('/signup');
  };

  return (
    <>
    <div
      id="photo"
      className="flex items-center justify-center h-full py-20"
    >
      <div className="w-full max-w-lg p-5 bg-white rounded-lg shadow-lg text-center">
        <div className="text-black text-5xl font-bold mb-4">
          Welcome to
        </div>
        <div className="text-black text-3xl font-semibold mb-6">
          Voting App
        </div>
        <div
          className="cursor-pointer inline-block"
          role="button"
          onClick={handleonclickdir}
        >
          <div className="text-black text-sm no-underline p-3 border-2 border-transparent rounded-md bg-gray-400 hover:bg-gray-500 transition duration-300">
            GET Started <i className="fa-solid fa-arrow-right"></i>
          </div>
        </div>
      </div>
    </div>
    </>
  );
}
