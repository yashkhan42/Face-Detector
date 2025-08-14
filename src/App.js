import React from "react";
import FaceDetector from "./components/FaceDetector";
import "./App.css";

function App() {
  return (
    <div className="App">
      <h1>Face Detector with Landmarks & Expressions</h1>
      <FaceDetector />
    </div>
  );
}

export default App;