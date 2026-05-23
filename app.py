"""
app.py — Face Shape Hairstyle Recommendation System
Run: python app.py
Folder structure:
  faceshape_final/
    app.py
    model/
      face_shape_model.pkl
      label_encoder.pkl
      model_meta.json
    static/
      index.html
Requirements: pip install flask numpy joblib opencv-python mediapipe
"""

import os, cv2, base64, json
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app = Flask(__name__, static_folder='static')

# ── Load model
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

model = joblib.load(os.path.join(MODEL_DIR, 'face_shape_model.pkl'))
le    = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

# ── FIX: Lowered OBLONG_MIN from 1.40 → 1.32 so more long faces are caught ──
# Old values: OBLONG_MIN=1.40, ROUND_MAX=1.15  ← too strict, Oblong never triggered
# New values: OBLONG_MIN=1.32, ROUND_MAX=1.12  ← balanced, all 3 classes reachable
OBLONG_MIN = 1.32   # was 1.40 — catches faces that are long but not extremely so
ROUND_MAX  = 1.12   # was 1.15 — slightly tighter to avoid misclassifying round faces

META_PATH = os.path.join(MODEL_DIR, 'model_meta.json')
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        meta = json.load(f)
    # NOTE: We intentionally OVERRIDE model_meta.json thresholds here
    # because the meta file has OBLONG_MIN=1.40 which is too strict.
    # The corrected values above take priority.
    print(f"ℹ️  model_meta.json found. Using corrected thresholds: "
          f"OBLONG_MIN={OBLONG_MIN}, ROUND_MAX={ROUND_MAX}")
else:
    print(f"ℹ️  model_meta.json not found. Using defaults: "
          f"OBLONG_MIN={OBLONG_MIN}, ROUND_MAX={ROUND_MAX}")

# ── MediaPipe setup 
try:
    # New API (mediapipe >= 0.10)
    from mediapipe.tasks import python as _mp_py
    from mediapipe.tasks.python import vision as _mp_vision
    _USE_NEW_API = True
except ImportError:
    _USE_NEW_API = False

# Also try legacy API as fallback
try:
    _mp_face_mesh_legacy = mp.solutions.face_mesh
    _USE_LEGACY = True
except AttributeError:
    _USE_LEGACY = False

# ── Hairstyle recommendations ─────────────────────────────────
HAIRSTYLES = {
    "Male": {
        "Oval": {
            "desc": "Lucky you — oval faces suit almost every hairstyle.",
            "tip": "Almost any haircut works. Experiment freely!",
            "styles": [
                {"name": "Textured Quiff",     "icon": "💈", "desc": "Volume on top, clean on sides"},
                {"name": "Side Part",           "icon": "✂️", "desc": "Classic, professional look"},
                {"name": "Undercut",            "icon": "⚡", "desc": "Bold contrast, modern edge"},
                {"name": "Caesar Cut",          "icon": "👑", "desc": "Short, textured, low maintenance"},
                {"name": "French Crop",         "icon": "🎯", "desc": "Short fringe, sharp lines"},
            ]
        },
        "Round": {
            "desc": "Add height and length to elongate the face visually.",
            "tip": "Avoid: buzz cuts, bowl cuts, or very wide styles.",
            "styles": [
                {"name": "Pompadour",           "icon": "🎸", "desc": "Height on top, sharp contrast"},
                {"name": "Faux Hawk",           "icon": "🦅", "desc": "Vertical volume, strong silhouette"},
                {"name": "High Fade + Texture", "icon": "💥", "desc": "Volume up top, tight on sides"},
                {"name": "Long Fringe",         "icon": "🌊", "desc": "Falls forward, slims the face"},
                {"name": "Angular Fringe",      "icon": "📐", "desc": "Sharp diagonal, adds angles"},
            ]
        },
        "Oblong": {
            "desc": "Add width and softness to balance the face length.",
            "tip": "Avoid: tall styles or long straight hair that adds length.",
            "styles": [
                {"name": "Medium Side Part",    "icon": "🎩", "desc": "Keeps length down, adds width"},
                {"name": "Textured Crop",       "icon": "🎲", "desc": "Short sides, textured top"},
                {"name": "Fringe / Bangs",      "icon": "🌿", "desc": "Horizontal line shortens face"},
                {"name": "Buzz Cut",            "icon": "⚙️",  "desc": "Minimizes length perception"},
                {"name": "Curtain Bangs",       "icon": "🪟", "desc": "Splits & widens face appearance"},
            ]
        }
    },
    "Female": {
        "Oval": {
            "desc": "The most versatile face shape — wear anything with confidence.",
            "tip": "You can pull off any length or style!",
            "styles": [
                {"name": "Layered Waves",       "icon": "🌊", "desc": "Effortless, textured movement"},
                {"name": "Blunt Bob",           "icon": "✂️", "desc": "Sharp, confident, modern"},
                {"name": "Curtain Bangs",       "icon": "🪟", "desc": "Soft framing, romantic feel"},
                {"name": "Long Straight",       "icon": "💫", "desc": "Elegant and sleek"},
                {"name": "Pixie Cut",           "icon": "⭐", "desc": "Bold and low maintenance"},
            ]
        },
        "Round": {
            "desc": "Elongate and define with strategic cuts that add length.",
            "tip": "Avoid: blunt bobs at chin level or very wide styles.",
            "styles": [
                {"name": "Long Layered Cut",    "icon": "🌿", "desc": "Draws eye downward, slims face"},
                {"name": "High Ponytail",       "icon": "🎀", "desc": "Adds vertical height"},
                {"name": "Side-Swept Bangs",    "icon": "💨", "desc": "Diagonal line, adds angles"},
                {"name": "Long Beach Waves",    "icon": "🏖️", "desc": "Below shoulders, very flattering"},
                {"name": "Razor-cut Layers",    "icon": "🔪", "desc": "Texture + movement + definition"},
            ]
        },
        "Oblong": {
            "desc": "Add softness and width to balance the face length.",
            "tip": "Avoid: center parts with very long straight hair.",
            "styles": [
                {"name": "Soft Waves",          "icon": "〰️", "desc": "Width + softness + bounce"},
                {"name": "Side Bangs",          "icon": "🌸", "desc": "Horizontal line breaks length"},
                {"name": "Shoulder-Length Bob", "icon": "💎", "desc": "Width at jawline level"},
                {"name": "Curly or Wavy Bob",   "icon": "🌀", "desc": "Volume on sides, shorter length"},
                {"name": "Blunt Bangs",         "icon": "📏", "desc": "Strong horizontal, very effective"},
            ]
        }
    }
}

# ── Feature extraction ────────────────────────────────────────
def _compute_ratios(pts, h, w):
    """Compute 9 facial ratios from landmark points."""
    def d(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    eps = 1e-6
    fl  = d(pts[10],  pts[152])
    fw  = d(pts[234], pts[454])
    jw  = d(pts[172], pts[397])
    ffw = d(pts[70],  pts[300])
    uf  = d(pts[10],  pts[168])
    lf  = d(pts[152], pts[164])

    feats = np.array([[
        fl / (fw + eps),
        jw / (fw + eps),
        ffw / (fw + eps),
        fl / (jw + eps),
        ffw / (jw + eps),
        fw / ((jw + ffw) / 2 + eps),
        uf / (fl + eps),
        lf / (fl + eps),
        jw / (ffw + eps)
    ]])
    r1 = fl / (fw + eps)
    return feats, float(r1)


def extract_features(image_np):
    """Extract 9 facial ratio features. Works with mediapipe 0.9.x and 0.10.x."""
    if image_np is None:
        return None, None

    # Convert to RGB
    if len(image_np.shape) == 2:
        rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    else:
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]

    # ── Try legacy API first (mediapipe < 0.10 or if solutions exists) ──
    if _USE_LEGACY:
        try:
            with _mp_face_mesh_legacy.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as fm:
                res = fm.process(rgb)
                if res.multi_face_landmarks:
                    lm  = res.multi_face_landmarks[0]
                    pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                    return _compute_ratios(pts, h, w)
        except Exception:
            pass  # fall through to new API

    # ── New Tasks API (mediapipe >= 0.10) ──────────────────────────────
    if _USE_NEW_API:
        try:
            import urllib.request, tempfile, os as _os
            model_url  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            model_path = _os.path.join(tempfile.gettempdir(), "face_landmarker.task")

            if not _os.path.exists(model_path):
                print("Downloading MediaPipe face landmarker model (~30 MB)…")
                urllib.request.urlretrieve(model_url, model_path)
                print("Downloaded.")

            BaseOptions    = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode     = mp.tasks.vision.RunningMode

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1
            )
            with FaceLandmarker.create_from_options(options) as lndmk:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = lndmk.detect(mp_image)

            if not result.face_landmarks:
                return None, None

            lm  = result.face_landmarks[0]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            return _compute_ratios(pts, h, w)

        except Exception as e:
            print(f"New API error: {e}")
            return None, None

    return None, None

# ── Routes
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Accept JSON with base64 image, return face shape + hairstyle data."""
    data = request.get_json(force=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    gender = data.get('gender', 'Male')
    if gender not in ('Male', 'Female'):
        gender = 'Male'

    try:
        # Decode base64 image
        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        img_bytes = base64.b64decode(img_b64)
        img_arr   = np.frombuffer(img_bytes, np.uint8)
        image     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image. Try a JPG or PNG file.'}), 400

        feats, r1 = extract_features(image)

        if feats is None:
            return jsonify({
                'error': 'No face detected. Please use a clear, front-facing photo with good lighting.'
            }), 422

        # ── FIXED: Improved Hybrid Prediction Logic ──────────────────────
        # Step 1: Always get ML prediction first
        ml_idx   = model.predict(feats)[0]
        ml_shape = le.inverse_transform([ml_idx])[0]

        # Step 2: Apply rule-based override for extreme r1 values
        #   OLD bug: OBLONG_MIN=1.40 was too strict → almost no one got Oblong
        #   FIX:     OBLONG_MIN=1.32 catches moderately long faces too
        if r1 > OBLONG_MIN:
            shape  = 'Oblong'
            method = 'rule (oblong override)'
        elif r1 < ROUND_MAX:
            shape  = 'Round'
            method = 'rule (round override)'
        else:
            # Middle zone: trust the ML model
            shape  = ml_shape
            method = 'ml'

        # Debug log — shows r1 and decision in terminal
        print(f"📐 r1={r1:.4f} | ML={ml_shape} | Final={shape} | Method={method}")

        # Confidence probabilities
        try:
            probs_raw  = model.predict_proba(feats)[0]
            confidence = {cls: round(float(p), 4) for cls, p in zip(le.classes_, probs_raw)}
        except Exception:
            confidence = {}

        info = HAIRSTYLES[gender][shape]

        return jsonify({
            'shape':      shape,
            'gender':     gender,
            'r1':         round(r1, 4),
            'method':     method,
            'confidence': confidence,
            'desc':       info['desc'],
            'tip':        info['tip'],
            'styles':     info['styles']
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("\n✅ ShapeSync server starting...")
    print(f"⚙️  Thresholds → OBLONG_MIN: {OBLONG_MIN} | ROUND_MAX: {ROUND_MAX}")
    print("📂 Model loaded from:", MODEL_DIR)
    print("🌐 Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)