import os
import cv2
import numpy as np
import time
import zipfile
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS

# === Google Drive ZIP setup ===
DRIVE_FILE_ID = "1CQSKhmsGqM7aI87suuiwSYDCH4NNfy2B"
ZIP_PATH = "new_features_npz.zip"
FEATURES_DIR = "new_features_npz"

# --- Download & extract once ---
if not os.path.exists(FEATURES_DIR):
    print("Downloading features from Google Drive using gdown...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)

    print("Validating zip file...")
    if not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError("Downloaded file is not a valid ZIP. Check your Google Drive file.")

    print("Extracting zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Features extracted successfully!\n")

# --- Config ---
MAX_DB_DESCRIPTORS = 5500
RATIO_THRESH = 0.65
orb = cv2.ORB_create(nfeatures=3500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# --- Lazy-load descriptors ---
available_locations = []
for file in os.listdir(FEATURES_DIR):
    if file.endswith("_features.npz"):
        loc = file.replace("_features.npz", "")
        available_locations.append(loc)
print(f"Available locations: {available_locations}")

db_cache = {}

def load_descriptors(loc):
    if loc in db_cache:
        return db_cache[loc]
    path = os.path.join(FEATURES_DIR, f"{loc}_features.npz")
    data = np.load(path, allow_pickle=True)
    des = data["descriptors"]
    if des.shape[0] > MAX_DB_DESCRIPTORS:
        des = des[:MAX_DB_DESCRIPTORS]
    db_cache[loc] = des
    return des

def recognize_location_from_image(img):
    start_time = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des_query = orb.detectAndCompute(gray, None)
    if des_query is None or len(des_query) == 0:
        return None, 0, time.time() - start_time

    best_loc, best_score = None, 0
    for loc in available_locations:
        des_db = load_descriptors(loc)
        matches = bf.knnMatch(des_query, des_db, k=2)
        good_matches = [
            m[0] for m in matches
            if len(m) == 2 and m[0].distance < RATIO_THRESH * m[1].distance
        ]
        score = len(good_matches)
        if score > best_score:
            best_score = score
            best_loc = loc

    return best_loc, best_score, time.time() - start_time

# --- Flask app ---
app = Flask(__name__)
CORS(app)

@app.route('/predict_location', methods=['POST'])
def predict_location():
    try:
        if not request.files:
            return jsonify({"error": "No image uploaded"}), 400

        images = [request.files[f"image{i}"] for i in range(10) if f"image{i}" in request.files]
        if not images:
            return jsonify({"error": "No image files detected"}), 400

        best_location, best_score, total_elapsed = None, 0, 0.0
        for idx, file in enumerate(images):
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if img is None:
                continue
            loc, score, elapsed = recognize_location_from_image(img)
            total_elapsed += elapsed
            if score > best_score:
                best_score = score
                best_location = loc
                
        print(f"Predicted location: {best_location or 'Unknown'} | Matches: {best_score} | Time: {round(total_elapsed, 2)}s")

        return jsonify({
            "predicted_location": best_location or "Unknown",
            "good_matches": int(best_score),
            "elapsed_time_sec": round(total_elapsed, 2)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
