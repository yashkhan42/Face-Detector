import React, { useRef, useState, useEffect } from "react";

const MODEL_URL = "https://justadudewhohacks.github.io/face-api.js/models";
const FACEAPI_CDN = "https://cdn.jsdelivr.net/npm/face-api.js/dist/face-api.min.js";
const MAX_WIDTH = 600; 

function loadFaceApiScript() {
  return new Promise((resolve, reject) => {
    if (window.faceapi) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = FACEAPI_CDN;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load face-api.js"));
    document.body.appendChild(script);
  });
}

function getMainExpression(expressions) {
  if (!expressions) return "";
  let max = 0;
  let main = "";
  Object.entries(expressions).forEach(([expr, val]) => {
    if (val > max) { max = val; main = expr; }
  });
  return main;
}

function drawLandmarkLine(ctx, points, close = false) {
  if (!points || points.length === 0) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  if (close) ctx.closePath();
  ctx.strokeStyle = "#0070f3";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function FaceDetector() {
  const [image, setImage] = useState(null);
  const [status, setStatus] = useState("Loading models...");
  const [faces, setFaces] = useState([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [displayDims, setDisplayDims] = useState({width: 0, height: 0});
  const canvasRef = useRef();
  const imgRef = useRef();

  useEffect(() => {
    let isMounted = true;
    const loadModels = async () => {
      try {
        await loadFaceApiScript();
        await window.faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
        await window.faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        await window.faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        if (isMounted) {
          setStatus("Models loaded. Upload an image!");
          setModelsLoaded(true);
        }
      } catch (err) {
        setStatus("Error loading face-api.js or models");
      }
    };
    loadModels();
    return () => { isMounted = false; };
  }, []);

  const handleImageChange = (e) => {
    setFaces([]);
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setImage(ev.target.result);
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  };

  useEffect(() => {
    const detectFaces = async () => {
      if (!image || !modelsLoaded) return;
      setStatus("Detecting faces...");
      const imgEl = imgRef.current;
      await new Promise((res) => {
        imgEl.onload = res;
        if (imgEl.complete && imgEl.naturalHeight !== 0) res();
      });

      let imgW = imgEl.naturalWidth;
      let imgH = imgEl.naturalHeight;
      let dispW = imgW;
      let dispH = imgH;
      if (imgW > MAX_WIDTH) {
        dispW = MAX_WIDTH;
        dispH = Math.round(imgH * (MAX_WIDTH / imgW));
      }
      setDisplayDims({width: dispW, height: dispH});

      const detections = await window.faceapi
        .detectAllFaces(imgEl)
        .withFaceLandmarks()
        .withFaceExpressions();

      setFaces(detections);
      const canvas = canvasRef.current;
      canvas.width = dispW;
      canvas.height = dispH;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imgEl, 0, 0, dispW, dispH);
      const scaleX = dispW / imgW;
      const scaleY = dispH / imgH;

      detections.forEach((det) => {
        const { x, y, width, height } = det.detection.box;
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 3;
        ctx.strokeRect(x * scaleX, y * scaleY, width * scaleX, height * scaleY);
        ctx.fillStyle = "#ff1744";
        det.landmarks.positions.forEach(pt => {
          ctx.beginPath();
          ctx.arc(pt.x * scaleX, pt.y * scaleY, 2, 0, 2*Math.PI);
          ctx.fill();
        });

        const lm = det.landmarks.positions;
        drawLandmarkLine(ctx, lm.slice(0, 17).map(pt => ({x: pt.x * scaleX, y: pt.y * scaleY})));      // Jaw
        drawLandmarkLine(ctx, lm.slice(17, 22).map(pt => ({x: pt.x * scaleX, y: pt.y * scaleY})));     // Right eyebrow
        drawLandmarkLine(ctx, lm.slice(22, 27).map(pt => ({x: pt.x * scaleX, y: pt.y * scaleY})));     // Left eyebrow
        drawLandmarkLine(ctx, lm.slice(27, 31).map(pt => ({x: pt.x * scaleX, y: pt.y * scaleY})));     // Nose bridge
        drawLandmarkLine(ctx, lm.slice(31, 36).map(pt => ({x: pt.x * scaleX, y: pt.y * scaleY})));     // Lower nose
        drawLandmarkLine(ctx, lm.slice(36, 42).map(pt => ({x: pt.x * scaleX, y: pt.y * scaleY})), true); // Right eye
        drawLandmarkLine(ctx, lm.slice(42, 48).map(pt => ({x: pt.x * scaleX, y: pt.y * scaleY})), true); // Left eye
        drawLandmarkLine(ctx, lm.slice(48, 60).map(pt => ({x: pt.x * scaleX, y: pt.y * scaleY})), true); // Outer lip
        drawLandmarkLine(ctx, lm.slice(60, 68).map(pt => ({x: pt.x * scaleX, y: pt.y * scaleY})), true); // Inner lip
        const expr = getMainExpression(det.expressions);
        if (expr) {
          ctx.font = "18px Arial";
          ctx.fillStyle = "#222";
          ctx.fillText(expr, x * scaleX, (y * scaleY) > 20 ? (y * scaleY) - 8 : (y * scaleY) + 20);
        }
      });

      setStatus(`Detected ${detections.length} face(s).`);
    };
    detectFaces();
  }, [image, modelsLoaded]);

  return (
    <div className="face-detector">
      <input type="file" accept="image/*" onChange={handleImageChange} disabled={!modelsLoaded} />
      <div className="preview-section">
        {image && (
          <>
            <img
              ref={imgRef}
              src={image}
              alt="Uploaded preview"
              style={{ display: "none" }}
              crossOrigin="anonymous"
            />
            <canvas
              ref={canvasRef}
              style={{
                maxWidth: "100%",
                width: `${displayDims.width}px`,
                height: `${displayDims.height}px`,
                borderRadius: "8px",
                boxShadow: "0 1px 8px rgba(0,0,0,0.05)"
              }}
            />
            {faces.length > 0 && (
              <div className="results">
                <h3>Faces & Expressions:</h3>
                <ul>
                  {faces.map((face, i) => (
                    <li key={i}>
                      <strong>Face {i+1}</strong>: {getMainExpression(face.expressions)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
      <p className="status">{status}</p>
    </div>
  );
}

export default FaceDetector;