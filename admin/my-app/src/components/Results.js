import React,{useContext,useEffect,useNavigate, useState} from 'react'
import Notecontext from "../context/notes/NotesContext";
import Resultitems from './Resultitems';

export default function Results() {
    const a = useContext(Notecontext);
    const navigate=useNavigate;
    const [winner,setwinner]=useState({name:"",voteCount:0});

    useEffect(() => {
        if (localStorage.getItem("token1")) {
          a.getNotes();
          findWinner();
          
        } else {
          navigate("/");
        }
        // eslint-disable-next-line
      }, []);
    
    //   {a.notes.forEach((note) => {
    //     if(note.voteCount>winner.voteCount){
    //         console.log(note.voteCount,note.name);
    //         setwinner(note.voteCount,note.name);
    //         return(null);
    //     }
    //   })}

  // Function to find the note with the highest votes
  const findWinner = () => {
    a.notes.forEach((note) => {
      if (note.voteCount > winner.voteCount) {
            setwinner({
                name:note.name,
                voteCount:note.voteCount
            })
      }
    });
  };
  return (
    <>
    <div className="container text-center" style={{height:"500px",display:"flex",justifyContent:"center",alignItems:"center"}}>
        <div className="row row-cols-auto" style={{display:"flex",justifyContent:"center",alignItems:"center"}}>
            {a.notes.map((note) => {
                return (
                    <Resultitems
                        key={note._id}
                        note={note}
                        name={note.name}
                        party={note.party}
                        voteCount={note.voteCount}
                    />
                );
            })}
        </div>
        
    </div>
    <div style={{display:'flex',justifyContent:"center",alignItems:"center",flexDirection:"column",textAlign: 'center',
    marginTop: '20px',
    padding: '20px',
    borderRadius: '10px',
    backgroundColor: '#f9f9f9',
    boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.1)',}}>
        
            <h2 style={{
                fontSize: '2rem',
                color: '#333',
                fontWeight: 'bold',
                marginBottom: '10px',
            }}>Winner</h2>
            <h3 style={{
                fontSize: '1.5rem',
                color: '#555',
                fontWeight: 'normal',
            }}>{winner.name} with {winner.voteCount} votes!</h3>
        
    </div>
    </>
  )
}
