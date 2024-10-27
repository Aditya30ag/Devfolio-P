import React from "react";

export default function Modal2(props) {
    document.body.style.color="black";
  return (
    <form className="container my-4">
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
        />
      </div>
      <div className="modal-footer mx-4">
              <button
                type="button"
                className="btn btn-secondary mx-2"
                onClick={props.handleonclickclose}
              >
                Close
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={props.handleonclick}
              >
                Add a New Candidate
              </button>
            </div>
    </form>
  );
}
