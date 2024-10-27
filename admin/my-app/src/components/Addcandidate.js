import React,{useContext,useState} from 'react'
import Modal2 from './Modal2'
import Notecontext from "../context/notes/NotesContext";
import { useNavigate } from "react-router-dom";

export default function Addcandidate() {
    const a = useContext(Notecontext);
    const navigate = useNavigate();
    const [notes, setnotes] = useState({
        name: "",
        party: "",
      });
    const onchange = (e) => {
        setnotes({ ...notes, [e.target.name]: e.target.value });
      };
    
    const handleonclick=async()=>{
        await a.addcandidate(notes.name, notes.party);
        navigate('/homeadmin')
    }
    const handleonclickclose=async(e)=>{
      navigate('/homeadmin')
  }
  return (
    <div>
        <Modal2 onchange={onchange} handleonclick={handleonclick} handleonclickclose={handleonclickclose}/>
    </div>
  )
}
