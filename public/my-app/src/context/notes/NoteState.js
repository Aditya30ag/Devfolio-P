import { useState } from "react";
import Notecontext from "./NotesContext";

const NoteState = (props) => {
  const host = "http://localhost:5000";
  const note = [];

  const [notes, setnotes] = useState(note);

  const getNotes = async () => {
    //Add api
    const response = await fetch(
      "http://localhost:5000/api/notes/fetchallcandidate",
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          token: localStorage.getItem("token"),
        },
      },
     
    );
    const json = await response.json();
    setnotes(json);
  };
  const updatecandidate = async (id, name, party, voteCount) => {
    let url = `${host}/api/notes/updatecandidate/${id}`;
    const response = await fetch(url, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name, party, voteCount }),
    });
    const json = await response.json();
    console.log(json);

    let newNotes = JSON.parse(JSON.stringify(notes));
    for (let i = 0; i < notes.length; i++) {
      const element = notes[i];
      if (element._id === id) {
        newNotes[i].name = name;
        newNotes[i].party = party;
        newNotes[i].voteCount = voteCount;
        break;
      }
    }
    //setnotes(newNotes);
  };
  //const ;
  const [user,setuser]=useState({});
  const getuser = async () => {
    //Add api
    const response = await fetch(
      "http://localhost:5000/api/auth/getuser",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          token: localStorage.getItem('token'),
        },
      });
    const json = await response.json();
    console.log(json)
    setuser(json)
  };
  
  const updateuser = async (id,name,email,aadharNumber,isvoted) => {
    let url = `${host}/api/auth/updateuser/${id}`;
    const response = await fetch(url, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        token: localStorage.getItem("token"),
      },
      body: JSON.stringify({name,email,aadharNumber,isvoted:true}),
    });
    const json = await response.json();
    console.log(json)
    setuser(json)
    /*let newNotes = JSON.parse(JSON.stringify(user));
    for (let i = 0; i < notes.length; i++) {
      const element = user[i];
      if (element._id === id) {
        newNotes[i].name = name;
        newNotes[i].email= email;
        newNotes[i].aadharNumber =aadharNumber;
        newNotes[i].isvoted =isvoted;
        break;
    }}
    setuser(newNotes);*/
  };
  const deletecandidate = async (id) => {
    //Add api
    const response = await fetch(
      `http://localhost:5000/api/notes/deletecandidate/${id}`,
      {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
          token1: localStorage.getItem('token1'),
        },
      });
    const json = await response.json();
    console.log(json)
  };
  const addcandidate=async(name,party)=>{
    const response = await fetch(
      `http://localhost:5000/api/notes/addcandidate`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          token1: localStorage.getItem('token1'),
        },
        body: JSON.stringify({ name,party}),
      })
      const json=await response.json();
      setnotes(notes.concat(json));
  }
  return (
    <Notecontext.Provider
      value={{ notes, getNotes, updatecandidate,user,getuser,updateuser,deletecandidate,addcandidate}}
    >
      {props.children}
    </Notecontext.Provider>
  );
};

export default NoteState;
