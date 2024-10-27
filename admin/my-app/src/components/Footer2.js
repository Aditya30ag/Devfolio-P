import React from 'react'
import { Link } from "react-router-dom";
export default function Footer() {
    
  return (
    <>
    
    <div style={{position:"absolute",bottom:"28px",marginTop:"0px",height:"35px",width:"100%",backgroundColor:"#d1e0e0",color:"black",display:"flex",justifyContent:"center",alignItems:"center",zIndex:"11"}}> 
        <Link style={{textDecoration:"none",fontSize:"10px",color:"black"}} to="/">Conditions of Use & Sale</Link>
        <Link style={{textDecoration:"none",marginLeft:"10px",fontSize:"10px",color:"black"}} to="/">Privacy Notice</Link>
        <Link style={{textDecoration:"none",marginLeft:"10px",fontSize:"10px",color:"black"}} to="/">Interest-Based Ads</Link>
    </div>
    <div style={{position:"absolute",bottom:"0px",height:"35px",width:"100%",backgroundColor:"#d1e0e0",display:"flex",justifyContent:"center",alignItems:"center",zIndex:"10"}}>
        <div style={{textDecoration:"none",fontSize:"10px",color:"black"}}><i className="fa-regular fa-copyright" style={{marginRight:"10px",color:"black"}}></i>Crops, Inc. or its affiliates</div>
    </div>
    </>
  )
}
