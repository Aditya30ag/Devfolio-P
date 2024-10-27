// import React from 'react';
// //import Notecontext from "../context/notes/NotesContext";
// //import { useNavigate } from "react-router-dom";

// export default function Noteitem(props) {
//   //const a = useContext(Notecontext);
//   const ab=()=>{
//     //e.preventDefault();
//     props.handleonclick(props.note);
//     //props.note
//   }
//   return (
//     <>
//     <div className="col-md-3">
//       <div className="card my-3" style={{backgroundColor:'#DED9DF',color:"black",width:"280px",height:"120px"}}>
//         <div className="card-body">
//           <h5 className="card-title">{props.name}</h5>
//           <p className="card-text">
//           {props.party}
//           </p>
//           <div className="d-flex justify-content-end"  >
//             <button className='btn btn-primary' onClick={ab} type="submit" disabled={props.disable} style={{position:"relative",bottom:"20px"}}>Vote</button>
//           </div>
//         </div>
//       </div>
//     </div>
//     </>
//   )
// }
import React from 'react';

export default function Noteitem(props) {
  const ab = () => {
    props.handleonclick(props.note);
  };

  return (
    <>
      <div className="col-md-3">
        <div className="card my-3 bg-gray-300 text-black" style={{ width: '280px', height: '120px' }}>
          <div className="card-body">
            <h5 className="card-title">{props.name}</h5>
            <p className="card-text">{props.party}</p>
            <div className="flex justify-end relative" style={{ bottom: '20px' }}>
              <button
                className="btn btn-primary"
                onClick={ab}
                type="submit"
                disabled={props.disable}
              >
                Vote
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
