import Navbar from "./components/Navbar";
import React, { useState } from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import LoadingBar from "react-top-loading-bar";
import Content from "./components/Content";
import Signup from "./components/Signup";
import About from "./components/About";
import NoteState from './context/notes/NoteState';
import End from "./components/End";
import Main from "./components/Main";
import './App.css';
import Footer from "./components/Footer";
import Footer2 from "./components/Footer2";
import Otp from "./components/Otp";
import Livecamera from "./components/Livecamera";
import WebcamCapture from "./components/Webcamcapture";

function App() {
  document.body.style.background="linear-gradient(#E2EAF4,#E2EAF4,white)";
  document.body.style.backgroundRepeat='no-repeat';
  const [progress, setprogress] = useState(0);
  const showalert = () => {
    setTimeout(() => {
      setprogress(20);
    }, 100);
    setTimeout(() => {
      setprogress(40);
    }, 200);
    setTimeout(() => {
      setprogress(60);
    }, 300);
    setTimeout(() => {
      setprogress(80);
    }, 400);
    setTimeout(() => {
      setprogress(100);
    }, 500);
  };
  const handleonClick2 = () => {
    showalert();
  };


  const router = createBrowserRouter([
    {
      path: "/",
      element: (
        <>
          <Navbar showalert={showalert} handleonClick2={handleonClick2} />
          <LoadingBar color="#000" progress={progress} />
          <Main/><Footer2/>
        </>
      ),
    },
    {
      path: "/home",
      element: (
        <>
          <Navbar showalert={showalert} handleonClick2={handleonClick2} />
          <LoadingBar color="#000" progress={progress} />
          <div className="container"><Content handleonClick2={handleonClick2}/></div><Footer2/>
        </>
      ),
    },
    {
      path: "/about",
      element: (
        <>
          <Navbar showalert={showalert} handleonClick2={handleonClick2} />
          <LoadingBar color="#000" progress={progress} />
          <About/><Footer/>
        </>
      ),
    },
    {
      path: "/otp",
      element: (
        <>
          <Navbar showalert={showalert} handleonClick2={handleonClick2} />
          <LoadingBar color="#000" progress={progress} />
          <Otp showalert={showalert}/><Livecamera/><Footer2/>
        </>
      ),
    },
    {
      path: "/webcamcapture",
      element: (
        <>
          <Navbar showalert={showalert} handleonClick2={handleonClick2} />
          <LoadingBar color="#000" progress={progress} />
          <WebcamCapture/><Footer2/>
        </>
      ),
    },
    {
      path: "/camera",
      element: (
        <>
          <Navbar showalert={showalert} handleonClick2={handleonClick2} />
          <LoadingBar color="#000" progress={progress} />
          <Livecamera showalert={showalert}/><Footer2/>
        </>
      ),
    },
    {
      path: "/signup",
      element: (
        <>
          <Navbar showalert={showalert} handleonClick2={handleonClick2} />
          <LoadingBar color="#000" progress={progress} />
          <div className='container'><Signup showalert={showalert} handleonClick2={handleonClick2} /></div><Footer2/>
        </>
      ),
    },
    {
      path: "/end",
      element: (
        <>
          <Navbar showalert={showalert} handleonClick2={handleonClick2} />
          <LoadingBar color="#000" progress={progress} />
          <div className='container'><End showalert={showalert} handleonClick2={handleonClick2}/></div><Footer2/>
        </>
      ),
    },
  ]);
  return (
    <>
    <NoteState>
    <RouterProvider router={router}/>
    </NoteState>
    </>
  )
}
export default App;
