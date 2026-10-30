const canvas = document.querySelector("#digit-canvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
const connectButton = document.querySelector("#connect");
const classifyButton = document.querySelector("#classify");
const clearButton = document.querySelector("#clear");
const prediction = document.querySelector("#prediction");
const statusText = document.querySelector("#status");
const statusDot = document.querySelector("#status-dot");

const FRAME_MAGIC = new Uint8Array([0x56, 0x47]);
let port;
let reader;
let readBuffer = "";
let drawing = false;
let previousPoint;
let pendingPrediction;

function clearCanvas() {
  context.fillStyle = "black";
  context.fillRect(0, 0, canvas.width, canvas.height);
  prediction.textContent = "—";
}

function canvasPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * canvas.width / bounds.width,
    y: (event.clientY - bounds.top) * canvas.height / bounds.height,
  };
}

function beginStroke(event) {
  drawing = true;
  previousPoint = canvasPoint(event);
  canvas.setPointerCapture(event.pointerId);
  context.beginPath();
  context.arc(previousPoint.x, previousPoint.y, 12, 0, Math.PI * 2);
  context.fillStyle = "white";
  context.fill();
}

function continueStroke(event) {
  if (!drawing) return;
  const point = canvasPoint(event);
  context.strokeStyle = "white";
  context.lineWidth = 24;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(previousPoint.x, previousPoint.y);
  context.lineTo(point.x, point.y);
  context.stroke();
  previousPoint = point;
}

function endStroke(event) {
  drawing = false;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
}

function setStatus(message, state = "") {
  statusText.textContent = message;
  statusDot.className = `status-dot ${state}`;
}

function makeTensor() {
  const source = context.getImageData(0, 0, canvas.width, canvas.height);
  let left = source.width;
  let top = source.height;
  let right = -1;
  let bottom = -1;

  for (let y = 0; y < source.height; y += 1) {
    for (let x = 0; x < source.width; x += 1) {
      if (source.data[(y * source.width + x) * 4] > 20) {
        left = Math.min(left, x);
        right = Math.max(right, x);
        top = Math.min(top, y);
        bottom = Math.max(bottom, y);
      }
    }
  }
  if (right < left) throw new Error("Draw a digit first");

  const cropWidth = right - left + 1;
  const cropHeight = bottom - top + 1;
  const scale = 20 / Math.max(cropWidth, cropHeight);
  const width = Math.max(1, Math.round(cropWidth * scale));
  const height = Math.max(1, Math.round(cropHeight * scale));
  const xOffset = Math.floor((28 - width) / 2);
  const yOffset = Math.floor((28 - height) / 2);
  const reduced = new OffscreenCanvas(28, 28);
  const reducedContext = reduced.getContext("2d", { willReadFrequently: true });
  reducedContext.fillStyle = "black";
  reducedContext.fillRect(0, 0, 28, 28);
  reducedContext.imageSmoothingEnabled = true;
  reducedContext.imageSmoothingQuality = "high";
  reducedContext.drawImage(canvas, left, top, cropWidth, cropHeight, xOffset, yOffset, width, height);

  const pixels = reducedContext.getImageData(0, 0, 28, 28).data;
  const tensor = new Uint8Array(784);
  for (let index = 0; index < tensor.length; index += 1) tensor[index] = pixels[index * 4];
  return tensor;
}

async function readSerial() {
  const decoder = new TextDecoder();
  reader = port.readable.getReader();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      readBuffer += decoder.decode(value, { stream: true });
      const lines = readBuffer.split(/\r?\n/);
      readBuffer = lines.pop() ?? "";
      for (const line of lines) handleLine(line.trim());
    }
  } catch (error) {
    if (port?.readable) setStatus(error.message);
  } finally {
    reader.releaseLock();
    reader = undefined;
  }
}

function handleLine(line) {
  if (line === "READY") {
    setStatus("Arduino ready", "online");
    classifyButton.disabled = false;
    return;
  }
  if (line.startsWith("P:") && /^P:[0-9]$/.test(line)) {
    prediction.textContent = line[2];
    setStatus("Inference complete", "online");
    classifyButton.disabled = false;
    pendingPrediction?.resolve(line[2]);
    pendingPrediction = undefined;
    return;
  }
  if (line.startsWith("E:")) {
    setStatus(`Arduino ${line.slice(2)} error`);
    classifyButton.disabled = false;
    pendingPrediction?.reject(new Error(line));
    pendingPrediction = undefined;
  }
}

async function connect() {
  if (!("serial" in navigator)) {
    setStatus("Use Chrome or Edge with Web Serial");
    return;
  }
  try {
    port = await navigator.serial.requestPort();
    await port.open({ baudRate: 57600 });
    connectButton.textContent = "Arduino connected";
    connectButton.disabled = true;
    classifyButton.disabled = true;
    setStatus("Connected — waiting for Arduino", "online");
    void readSerial();
  } catch (error) {
    setStatus(error.message);
  }
}

async function classify() {
  try {
    const tensor = makeTensor();
    const frame = new Uint8Array(FRAME_MAGIC.length + tensor.length + 1);
    frame.set(FRAME_MAGIC, 0);
    frame.set(tensor, FRAME_MAGIC.length);
    frame[frame.length - 1] = tensor.reduce((sum, value) => (sum + value) & 0xff, 0);

    classifyButton.disabled = true;
    prediction.textContent = "·";
    setStatus("Sending 784 pixels…", "busy");
    const response = new Promise((resolve, reject) => {
      pendingPrediction = { resolve, reject };
      setTimeout(() => {
        if (pendingPrediction) {
          pendingPrediction = undefined;
          reject(new Error("Inference timed out"));
        }
      }, 5000);
    });
    const writer = port.writable.getWriter();
    try {
      await writer.write(frame);
    } finally {
      writer.releaseLock();
    }
    await response;
  } catch (error) {
    pendingPrediction = undefined;
    classifyButton.disabled = !port;
    prediction.textContent = "—";
    setStatus(error.message);
  }
}

canvas.addEventListener("pointerdown", beginStroke);
canvas.addEventListener("pointermove", continueStroke);
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);
clearButton.addEventListener("click", clearCanvas);
connectButton.addEventListener("click", connect);
classifyButton.addEventListener("click", classify);
navigator.serial?.addEventListener("disconnect", () => {
  port = undefined;
  connectButton.disabled = false;
  connectButton.textContent = "Connect Arduino";
  classifyButton.disabled = true;
  setStatus("Arduino disconnected");
});

clearCanvas();
