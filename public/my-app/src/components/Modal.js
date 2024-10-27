import React from "react";

export default function Modal(props) {
    document.body.style.color="black";
  return (
    <>
    <form>
      <div className="mb-3">
        <label htmlFor="title" className="form-label">
          Name
        </label>
        <input
          type="text"
          className="form-control"
          id="name"
          name="name"
          onChange={props.onchange}
          value={props.notes.name}
        />
      </div>
      <div className="mb-3">
        <label htmlFor="party" className="form-label">
          party
        </label>
        <input
          type="text"
          className="form-control"
          id="party"
          name="party"
          onChange={props.onchange}
          value={props.notes.party}
        />
      </div>
    </form>
    </>
  );
}
