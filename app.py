from flask import Flask, request, render_template_string
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = joblib.load("savedmodel.pth")

html = """
<!doctype html>
<title>Olivetti Face Classifier</title>
<h2>Upload a 64x64 grayscale image</h2>
<form method="POST" enctype="multipart/form-data">
  <input type="file" name="image" />
  <input type="submit" value="Upload" />
</form>

{% if pred is not none %}
<h3>Predicted Class: {{ pred }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None

    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file).convert("L").resize((64, 64))
        arr = np.array(img).reshape(1, -1)
        pred = int(model.predict(arr)[0])

    return render_template_string(html, pred=pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
