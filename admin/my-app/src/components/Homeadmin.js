// import React, { useEffect, useContext, useRef, useState } from "react";
// import { Link, useNavigate } from "react-router-dom";
// import Noteitem1 from "./Noteitem1";
// import Notecontext from "../context/notes/NotesContext";
// import Modal from "./Modal";
// import "./Homeadmin.css";

// export default function Homeadmin(props) {
//   const ref = useRef("null");
//   const refclose = useRef("null");
//   const a = useContext(Notecontext);
//   const navigate = useNavigate();

//   useEffect(() => {
//     if (localStorage.getItem("token1")) {
//       a.getNotes();
//     } else {
//       navigate("/");
//     }
//     // eslint-disable-next-line
//   }, []);
//   const [notes, setnotes] = useState({
//     id: "",
//     name: "",
//     party: "",
//   });
//   const onchange = (e) => {
//     setnotes({ ...notes, [e.target.name]: e.target.value });
//   };
//   const updatenote = (note) => {
//     ref.current.click();
//     setnotes({
//       id: note._id,
//       name: note.name,
//       party: note.party,
//     });
//   };
//   const handleonClick = async(e) => {
//     e.preventDefault();
//     await a.updatecandidate(notes.id, notes.name, notes.party, notes.voteCount);
//     window.location.reload();
//     refclose.current.click();
//     props.handleonClick2();
//   };
//   const signuplogout=()=>{
//     localStorage.removeItem("token1");
//   }
//   return (
//     <>
//       <button
//         type="button"
//         className="btn btn-primary"
//         data-bs-toggle="modal"
//         data-bs-target="#exampleModal"
//         ref={ref}
//         style={{ display: "none" }}
//       >
//         Update a Note
//       </button>
//       <div
//         className="modal fade"
//         id="exampleModal"
//         tabIndex="-1"
//         aria-labelledby="exampleModalLabel"
//         aria-hidden="true"
//         style={{ marginTop: "100px" }}
//       >
//         <div className="modal-dialog">
//           <div
//             className="modal-content"
//             style={{ background: "linear-gradient(#E2EAF4,white)" }}
//           >
//             <div className="modal-header">
//               <h1 className="modal-title fs-5" id="exampleModalLabel">
//                 Update a Candidate
//               </h1>
//               <button
//                 type="button"
//                 className="btn-close"
//                 data-bs-dismiss="modal"
//                 aria-label="Close"
//               ></button>
//             </div>
//             <div className="modal-body" id="exampleModal">
//               <Modal notes={notes} onchange={onchange} />
//             </div> 
//             <div className="modal-footer">
//               <button
//                 type="button"
//                 className="btn btn-secondary"
//                 data-bs-dismiss="modal"
//                 ref={refclose}
//               >
//                 Close
//               </button>
//               <button
//                 type="button"
//                 className="btn btn-primary"
//                 onClick={handleonClick}
//               >
//                 Save changes
//               </button>
//             </div>
//           </div>
//         </div>
//       </div>
//       <div
//         className="container my-4"
//         style={{
//           display: "flex",
//           padding: "10px 10px 10px 10px",
//           flexDirection:"row",
//           columnGap: "100px",
//         }}
//       >
//         <div
//           className="container"
//           style={{
//             display: "flex",
//             justifyContent:"left",
//             alignItems: "center",
//             flexDirection: "column",
//           }}
//         >
//           <div>
//           {a.notes.map((note) => {
//             return (
//               <Noteitem1
//                 key={note._id}
//                 note={note}
//                 name={note.name}
//                 party={note.party}
//                 updatenote={updatenote}
//               />
//             );
//           })}
//           </div>
//           <button className="btn btn-success"><Link to="/addcandidate" style={{textDecoration:'none',color:"whitesmoke"}}>Add a New Candidate</Link></button>
//           <button className="btn btn-success" style={{marginTop:"10px"}}><Link to="/results" style={{textDecoration:'none',color:"whitesmoke"}}>View Results</Link></button>
//         </div>
//         <div className="imgstyle" style={{ rowGap: "10px",textAlign:"center"}}>
//           <img
//             height="200px"
//             style={{ backgroundSize: "cover", padding: "10px" }}
//             src="Screenshot 2024-07-13 160018.png"
//             alt="error"
//           />
//           <img
//             height="200px"
//             style={{ backgroundSize: "cover", padding: "10px" }}
//             src="Screenshot 2024-07-13 160936.png"
//             alt="error"
//           />
//           <img
//             height="200px"
//             style={{ backgroundSize: "cover", padding: "10px" }}
//             src="Screenshot 2024-07-13 161551.png"
//             alt="error"
//           />
//           <img
//             height="400px"
//             style={{ backgroundSize: "cover", padding: "10px" }}
//             src="Screenshot 2024-07-13 160104.png"
//             alt="error"
//           />
//           <div><Link to="/signupadmin" style={{textDecoration:"none"}} onClick={signuplogout}>Add a new admin user</Link></div>
//         </div>
//       </div>
//     </>
//   );
// }
import React, { useEffect, useContext, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Noteitem1 from "./Noteitem1";
import Notecontext from "../context/notes/NotesContext";

export default function Homeadmin(props) {
  
  const a = useContext(Notecontext);
  const navigate = useNavigate();

  useEffect(() => {
    if (localStorage.getItem("token1")) {
      a.getNotes();
    } else {
      navigate("/");
    }
  }, [a, navigate]);

  const [notes, setnotes] = useState({
    id: "",
    name: "",
    party: "",
  });

  const onchange = (e) => {
    setnotes({ ...notes, [e.target.name]: e.target.value });
  };

  const updatenote = (note) => {
    setnotes({
      id: note._id,
      name: note.name,
      party: note.party,
    });
    setShowWeather(!showWeather);
  };

  const handleonClick = async (e) => {
    e.preventDefault();
    await a.updatecandidate(notes.id, notes.name, notes.party);
    window.location.reload();
    props.handleonClick2();
  };
  const close=(()=>{
    setShowWeather(!showWeather);
  })
  const signuplogout = () => {
    localStorage.removeItem("token1");
  };

  const [showWeather, setShowWeather] = useState(false);
  const [viewresult, setviewresult] = useState(false);
  const [value,setvalue]=useState("");
  const on = (e) => {
    setviewresult(false);
    setvalue(e.target.value)
  };
  const res=((e)=>{
    e.preventDefault();
    setTimeout(() => {
      setviewresult(true);
    }, value);
  })
  
  return (
    <>
    
    {showWeather && (
      <div className="w-full max-w-lg p-5 bg-white rounded-lg shadow-lg text-center" style={{position:"absolute",top:'25%',left:"30%"}}>
      <div>
        <form className="container my-4">
            <div className="mb-3">
              <label htmlFor="title" className="text-black text-3xl font-semibold mb-6">
                Name
              </label>
              <input
                type="text"
                className="form-control"
                id="name"
                name="name"
                onChange={onchange}
                value={notes.name}
              />
            </div>
            <div className="mb-3">
              <label htmlFor="party" className="text-black text-3xl font-semibold mb-6">
                party
              </label>
              <input
                type="text"
                className="form-control"
                id="party"
                name="party"
                onChange={onchange}
                value={notes.party}
              />
            </div>
            <div className="modal-footer mx-4">
                <button
                  type="button"
                  className="btn btn-secondary mx-2"
                  onClick={close}
                >
                  Close
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={handleonClick}
                >
                  Update Candidate
                </button>
              </div>
          </form>
      </div>
      </div>
    )}
    
      {/* Main content container */}
      <div className="container mx-auto px-4 my-10 flex flex-col md:flex-row justify-between space-y-8 md:space-y-0">
        <div className="flex flex-col space-y-4">
          {a.notes.map((note) => {
            return (
              <Noteitem1
                key={note._id}
                note={note}
                name={note.name}
                party={note.party}
                updatenote={updatenote}
              />
            );
          })}
          <Link
            to="/addcandidate"
            className="inline-block w-full px-4 py-2 mt-4 text-center text-white bg-green-600 rounded hover:bg-green-700"
          >
            Add a New Candidate
          </Link>
          <div >
              <form className="bg-white shadow-md rounded-lg p-6 max-w-xs">
                <label htmlFor="time" className="block text-sm font-medium text-gray-700 mb-2">
                  Set Voting Duration
                </label>
                <input
                  type="text"
                  id="time"
                  onChange={on}
                  className="border border-gray-300 rounded-lg p-2 mb-4 w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter duration (e.g., 30 minutes)"
                />
                <button
                  type="button"
                  onClick={res}
                  className="w-full bg-blue-600 text-white font-semibold py-2 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
                >
                  Set Duration
                </button>
              </form>
            </div>
          {viewresult &&
          <Link
            to="/results"
            className="inline-block w-full px-4 py-2 mt-4 text-center text-white bg-indigo-600 rounded hover:bg-indigo-700"
          >
            View Results
          </Link>}
        </div>

        {/* Image Section in Grid Format */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 gap-4" style={{marginTop:"20px",marginLeft:"20px"}}>
          <img
            className="w-full h-64 object-cover rounded-lg"
            src="Screenshot 2024-07-13 160018.png"
            alt="Screenshot 1"
          />
          <img
            className="w-full h-64 object-cover rounded-lg"
            src="Screenshot 2024-07-13 160936.png"
            alt="Screenshot 2"
          />
          <img
            className="w-full h-64 object-cover rounded-lg"
            src="Screenshot 2024-07-13 161551.png"
            alt="Screenshot 3"
          />
          <img
            className="w-full h-64 object-cover rounded-lg"
            src="Screenshot 2024-07-13 160104.png"
            alt="Screenshot 4"
          />
        
          <Link
            to="/signupadmin"
            className="col-span-2 block text-center text-blue-600 hover:underline"
            onClick={signuplogout}
          >
            Add a new admin user
          </Link>
        </div>
      </div>
    </>
  );
}

