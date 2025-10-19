import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

// ---------------------
// Global variables
// ---------------------
let classifier = null;
let modelReady = false;
const LABELS = [
  "biodegradable waste such as food scraps, leaves, or vegetable peels",
  "non biodegradable waste such as plastic, glass, or metal",
  "recyclable waste such as paper, cardboard, or bottles",
  "organic waste like fruit or vegetable matter",
  "plastic waste like bottles, wrappers, or bags",
  "metal waste such as cans, wires, or tools",
  "paper waste such as newspapers or cartons",
  "glass waste such as jars, bulbs, or bottles"
];

// DOM elements
const loader = document.getElementById("loader");
const preview = document.getElementById("preview");
const outputBox = document.getElementById("outputBox");
const uploadInput = document.getElementById("uploadInput");
const captureBtn = document.getElementById("captureBtn");

// ---------------------
// Model Loading
// ---------------------
async function loadModel() {
  loader.textContent = "Loading model (may take 10–20s)...";
  classifier = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32");
  loader.textContent = "Model loaded! You can upload or capture images now.";
  modelReady = true;
}
loadModel();

// ---------------------
// Image Classification
// ---------------------
async function classifyImage(imgEl) {
  if (!modelReady) {
    loader.textContent = "Model is still loading...";
    return;
  }
  loader.textContent = "Classifying...";
  hideOutput();

  try {
    const result = await classifier(imgEl, LABELS);
    displayResult(result);
  } catch (err) {
    loader.textContent = "Error during classification: " + err.message;
    console.error(err);
  }
}

// ---------------------
// Display Functions
// ---------------------
function displayResult(result) {
  const top = result[0];
  const response = {
    predicted_label: top.label,
    confidence: Number(top.score.toFixed(4)),
    all_scores: Object.fromEntries(result.map(r => [r.label, Number(r.score.toFixed(4))])),
    timestamp: new Date().toISOString()
  };
  outputBox.textContent = JSON.stringify(response, null, 2);
  outputBox.classList.remove("hidden");
  loader.textContent = "Done.";
}

function hideOutput() {
  outputBox.classList.add("hidden");
  outputBox.textContent = "";
}

// ---------------------
// File Upload Handler
// ---------------------
uploadInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (ev) => {
    preview.src = ev.target.result;
    preview.classList.remove("hidden");
    classifyImage(preview);
  };
  reader.readAsDataURL(file);
});

// ---------------------
// Webcam Capture Handler
// ---------------------
captureBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const video = document.createElement("video");
    video.srcObject = stream;
    await video.play();

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    stream.getTracks().forEach(t => t.stop());
    preview.src = canvas.toDataURL("image/png");
    preview.classList.remove("hidden");
    classifyImage(preview);
  } catch (err) {
    alert("Camera access failed: " + err.message);
    console.error(err);
  }
});
