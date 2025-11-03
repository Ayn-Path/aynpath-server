import os
import cv2
import numpy as np
import time
import requests
import zipfile
from flask import Flask, request, jsonify

# It will download features_npz from Google Drive
DRIVE_ZIP_URL = "https://drive.google.com/file/d/12jW_xT_ukUGa4TO5UL54prg6GP_ja-8c/view?usp=sharing"
ZIP_PATH = "features_npz.zip"
FEATURES_DIR = "features_npz"

if not os.path.exists(FEATURES_DIR):
    print("Downloading features from Google Drive...")
    r = requests.get(DRIVE_ZIP_URL, stream=True)
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Extracting zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Features extracted successfully!\n")

# --- Configuration ---
MAX_DB_DESCRIPTORS = 200000
RATIO_THRESH = 0.75

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# --- Load database ---
db_features = {}
for file in os.listdir(FEATURES_DIR):
    if file.endswith("_features.npz"):
        loc = file.replace("_features.npz", "")
        try:
            data = np.load(os.path.join(FEATURES_DIR, file), allow_pickle=True)
            descriptors = data["descriptors"]

            if descriptors.shape[0] > MAX_DB_DESCRIPTORS:
                descriptors = descriptors[:MAX_DB_DESCRIPTORS]

            db_features[loc] = descriptors
            print(f"Loaded {loc}: {descriptors.shape[0]} descriptors")

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

print("\nDatabase loaded successfully!")
print(f"Total locations: {len(db_features)}\n")

# --- Flask App ---
app = Flask(__name__)

def recognize_location_from_image(img):
    start_time = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des_query = orb.detectAndCompute(gray, None)

    if des_query is None or len(des_query) == 0:
        return None, 0, time.time() - start_time

    best_loc, best_score = None, 0
    for loc, des_db in db_features.items():
        matches = bf.knnMatch(des_query, des_db, k=2)
        good_matches = [m_n[0] for m_n in matches if len(m_n) == 2 and m_n[0].distance < RATIO_THRESH * m_n[1].distance]
        score = len(good_matches)
        if score > best_score:
            best_score = score
            best_loc = loc

    return best_loc, best_score, time.time() - start_time

@app.route('/predict_location', methods=['POST'])
def predict_location():
    images = [f for f in request.files if f.startswith("image")]
    if not images:
        return jsonify({"error": "No image uploaded"}), 400

    best_location, best_score, total_elapsed = None, 0, 0.0
    try:
        for key in images:
            file = request.files[key]
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if img is None:
                continue
            loc, score, elapsed = recognize_location_from_image(img)
            total_elapsed += elapsed
            if score > best_score:
                best_score = score
                best_location = loc

        response = {
            "predicted_location": best_location or "Unknown",
            "good_matches": best_score,
            "elapsed_time_sec": round(total_elapsed, 2)
        }
        print(f"Prediction: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)
