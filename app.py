from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
from PIL import Image
import io
import base64
import traceback

app = Flask(__name__)

# Load your trained model file (must be in same folder)
MODEL_PATH = "savedmodel.pth"
model = joblib.load(MODEL_PATH)

# HTML template (single-file). Uses Bootstrap CDN.
HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Olivetti Face Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { background:#f6f7fb; }
      #drop-area { background: #ffffff; transition: box-shadow .15s ease; border: 1px dashed #ddd; border-radius: 8px; padding: 20px; }
      #drop-area.dragover { box-shadow: 0 8px 30px rgba(0,0,0,0.08); border-color: #b3d7ff; }
      #previewImg { max-width:100%; max-height:320px; display:block; margin:0 auto; border-radius:6px; }
      .card { border-radius: 12px; }
      .spinner-container { min-height: 2.4rem; display:flex; align-items:center; }
    </style>
  </head>
  <body>
    <div class="container py-5">
      <div class="row justify-content-center">
        <div class="col-md-8">
          <div class="card shadow-sm">
            <div class="card-body">
              <h3 class="card-title mb-1">Olivetti Face Classifier</h3>
              <p class="text-muted small mb-3">Upload a face image (client resizes to 64×64). Predicts class using saved DecisionTree model.</p>

              <div id="drop-area" class="text-center">
                <p class="mb-1">Drag & drop an image here, or</p>
                <input id="fileElem" type="file" accept="image/*" style="display:none">
                <button id="fileSelect" class="btn btn-primary btn-sm">Choose a file</button>
                <div id="preview" class="mt-3">
                  <img id="previewImg" src="" alt="" class="d-none" />
                </div>
              </div>

              <div class="mt-3 d-flex justify-content-between align-items-center">
                <div>
                  <button id="predictBtn" class="btn btn-success" disabled>Predict</button>
                  <button id="resetBtn" class="btn btn-secondary btn-sm">Reset</button>
                </div>
                <div class="spinner-container">
                  <div id="spinner" class="d-none">
                    <div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>
                  </div>
                </div>
              </div>

              <div id="resultCard" class="mt-4 d-none">
                <h5>Result</h5>
                <div id="resultMain" class="mb-2"></div>
                <div id="probList"></div>
              </div>

              <hr />
              <small class="text-muted">This single-file app contains UI & server — drop savedmodel.pth here and run <code>python app.py</code>.</small>
            </div>
          </div>

          <div class="mt-3 text-center small text-muted">
            <span>Model: DecisionTreeClassifier | Dataset: Olivetti faces</span>
          </div>
        </div>
      </div>
    </div>

    <script>
      const fileSelect = document.getElementById("fileSelect");
      const fileElem = document.getElementById("fileElem");
      const dropArea = document.getElementById("drop-area");
      const previewImg = document.getElementById("previewImg");
      const predictBtn = document.getElementById("predictBtn");
      const resetBtn = document.getElementById("resetBtn");
      const spinner = document.getElementById("spinner");
      const resultCard = document.getElementById("resultCard");
      const resultMain = document.getElementById("resultMain");
      const probList = document.getElementById("probList");

      let imageBlob = null;

      fileSelect.addEventListener("click", e => fileElem.click());
      fileElem.addEventListener("change", handleFiles);

      ;["dragenter","dragover","dragleave","drop"].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
      });

      function preventDefaults(e){
        e.preventDefault();
        e.stopPropagation();
      }

      dropArea.addEventListener("dragenter", ()=> dropArea.classList.add("dragover"));
      dropArea.addEventListener("dragleave", ()=> dropArea.classList.remove("dragover"));
      dropArea.addEventListener("drop", (e) => {
        dropArea.classList.remove("dragover");
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles({ target: { files }});
      });

      function handleFiles(e){
        const files = e.target.files;
        if(!files || files.length === 0) return;
        const file = files[0];
        if(!file.type.startsWith("image/")) return alert("Please select an image file");
        previewFile(file);
      }

      function previewFile(file){
        const reader = new FileReader();
        reader.onload = function(event){
          const img = new Image();
          img.onload = function(){
            previewImg.src = event.target.result;
            previewImg.classList.remove("d-none");

            // create 64x64 offscreen canvas for model input
            const off = document.createElement("canvas");
            off.width = 64;
            off.height = 64;
            const offCtx = off.getContext("2d");
            // draw image into 64x64 box (preserves aspect by stretching - OK for this dataset)
            offCtx.drawImage(img, 0, 0, 64, 64);
            // convert to blob for upload
            off.toBlob(function(blob){
              imageBlob = blob;
              predictBtn.disabled = false;
            }, "image/png");
          };
          img.src = event.target.result;
        };
        reader.readAsDataURL(file);
      }

      predictBtn.addEventListener("click", async () => {
        if(!imageBlob) return;
        resultCard.classList.add("d-none");
        spinner.classList.remove("d-none");
        predictBtn.disabled = true;

        const fd = new FormData();
        fd.append("file", imageBlob, "upload.png");

        try {
          const res = await fetch("/predict", {
            method: "POST",
            body: fd
          });
          const data = await res.json();
          spinner.classList.add("d-none");
          predictBtn.disabled = false;
          if(data.error){
            resultMain.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            resultCard.classList.remove("d-none");
            return;
          }
          resultMain.innerHTML = `<h4>Predicted Class: <span class="badge bg-primary">${data.predicted_class}</span></h4>`;
          probList.innerHTML = "";
          if(data.top_predictions && data.top_predictions.length){
            const list = document.createElement("ul");
            list.className = "list-group";
            data.top_predictions.forEach(p => {
              const li = document.createElement("li");
              li.className = "list-group-item d-flex justify-content-between align-items-center";
              li.innerHTML = `<span>Class ${p.class}</span><span class="badge bg-info text-dark">${(p.prob*100).toFixed(2)}%</span>`;
              list.appendChild(li);
            });
            probList.appendChild(list);
          } else {
            probList.innerHTML = `<div class="text-muted">No probability info available from model.</div>`;
          }
          resultCard.classList.remove("d-none");
        } catch(err){
          spinner.classList.add("d-none");
          predictBtn.disabled = false;
          resultMain.innerHTML = `<div class="alert alert-danger">Request failed: ${err}</div>`;
          resultCard.classList.remove("d-none");
        }
      });

      resetBtn.addEventListener("click", () => {
        previewImg.src = "";
        previewImg.classList.add("d-none");
        imageBlob = null;
        predictBtn.disabled = true;
        resultCard.classList.add("d-none");
        probList.innerHTML = "";
        resultMain.innerHTML = "";
      });
    </script>
  </body>
</html>
"""

def preprocess_image_bytes(image_bytes):
    """Convert bytes to grayscale 64x64 flattened array"""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((64,64))
    arr = np.array(img).reshape(1, -1).astype(np.float32)
    return arr

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            f = request.files["file"]
            image_bytes = f.read()
        else:
            # allow base64 JSON fallback
            data = request.get_json(silent=True)
            if not data or "imageBase64" not in data:
                return jsonify({"error": "No image provided"}), 400
            header, b64 = data["imageBase64"].split(",",1) if "," in data["imageBase64"] else ("", data["imageBase64"])
            image_bytes = base64.b64decode(b64)

        X = preprocess_image_bytes(image_bytes)

        pred = int(model.predict(X)[0])

        probs = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
                top_idx = np.argsort(proba)[::-1][:3]
                probs = [{"class": int(i), "prob": float(round(float(proba[i]), 4))} for i in top_idx]
            except Exception:
                probs = None

        return jsonify({"predicted_class": pred, "top_predictions": probs})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}", "trace": tb}), 500

if __name__ == "__main__":
    # Run with debug=False for production-like behavior
    app.run(host="0.0.0.0", port=5000, debug=False)
