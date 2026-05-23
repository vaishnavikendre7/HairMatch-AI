# HairStyleAI — Face Shape Hairstyle Advisor
## Setup in 4 Steps

### 1. Project structure must be exactly this:
```
faceshape_final/
  app.py
  requirements.txt
  model/
    face_shape_model.pkl     ← copy from your downloaded Colab files
    label_encoder.pkl        ← copy from your downloaded Colab files
    model_meta.json          ← already included
  static/
    index.html               ← already included
```

### 2. Install dependencies
```bash
pip install flask numpy joblib opencv-python mediapipe
```

### 3. Copy your model files into model/ folder
- `face_shape_model.pkl`
- `label_encoder.pkl`
- Keep `model_meta.json` as-is (already present)

### 4. Run the server
```bash
python app.py
```
Open browser: **http://localhost:5000**

---
## What was fixed vs the old version

| Problem | Old Code | Fixed |
|---------|----------|-------|
| Frontend sends JSON, backend reads form data | `request.form['r1']` | `request.get_json()` with base64 image |
| No image processing in old app.py | Manual r1-r9 form fields | MediaPipe auto-extracts features from uploaded photo |
| Templates folder mismatch | `render_template('index.html')` looking in templates/ | `send_from_directory('static','index.html')` |
| No gender-based recommendations | Same hairstyles for all | Separate Male/Female style lists |

---
## API
`POST /predict`
- Body: `{ "image": "<base64 data URL>", "gender": "Male" | "Female" }`
- Returns: `{ shape, desc, tip, styles, confidence, r1, method }`
