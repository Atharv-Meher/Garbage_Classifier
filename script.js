import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.9.0";

// DOM elements
const camera = document.getElementById("camera");
const canvas = document.getElementById("snapshot");
const captureBtn = document.getElementById("captureBtn");
const imageUpload = document.getElementById("imageUpload");
const preview = document.getElementById("preview");
const output = document.getElementById("output");

// Garbage categories
const labels = [
  "biodegradable waste",
  "recyclable waste",
  "non-recyclable waste"
];

// Initialize camera
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    camera.srcObject = stream;
  } catch (err) {
    console.warn("Camera unavailable or blocked.");
    camera.style.display = "none";
  }
}

// Load CLIP model once
let classifierPromise = pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32");

// --- Capture from camera ---
captureBtn.addEventListener("click", async () => {
  if (camera.style.display === "none") {
    alert("Camera not accessible. Try uploading an image instead.");
    return;
  }

  const context = canvas.getContext("2d");
  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  context.drawImage(camera, 0, 0, canvas.width, canvas.height);

  const imgURL = canvas.toDataURL("image/jpeg");
  preview.src = imgURL;
  await classify(imgURL);
});

// --- Upload from file ---
imageUpload.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const imgURL = URL.createObjectURL(file);
  preview.src = imgURL;
  await classify(imgURL);
});

// --- Classification logic ---
async function classify(imgURL) {
  output.textContent = "Classifying...";
  const classifier = await classifierPromise;
  const result = await classifier(imgURL, labels.map(l => `photo of ${l}`));
  const best = result[0];
  const jsonOutput = {
    predicted_label: best.label.replace("photo of ", ""),
    confidence: Number(best.score.toFixed(4)),
    timestamp: new Date().toISOString()
  };
  output.textContent = JSON.stringify(jsonOutput, null, 4);
}

initCamera();
