import React from "react";
import "./noteitem.css";

export default function Noteitem(props) {

  return (
    <>
      <div >
        <div
          className="card my-3 cardcon"
          style={{
            backgroundColor: "#DED9DF",
            color: "black",
            width: "280px",
            height: "120px",
          }}
        >
          <div className="card-body">
            <h5 className="card-title">{props.note.name}</h5>
            <p className="card-text">{props.note.party}</p>
            <h5 className="card-title" style={{fontWeight:"bold"}}>Vote Count: {props.note.voteCount}</h5>
          </div>
        </div>
      </div>
    </>
  );
}
