import os
import cv2
import numpy as np
import time
from flask import Flask, request, jsonify

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "features_npz")
MAX_DB_DESCRIPTORS = 200000  # limit per location for faster matching
RATIO_THRESH = 0.75           # Lowe's ratio test threshold

# ORB detector and BFMatcher
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Load Database
db_features = {}
for file in os.listdir(FEATURES_DIR):
    if file.endswith("_features.npz"):
        loc = file.replace("_features.npz", "")
        # Allow pickle because descriptors might be stored as pickled objects
        data = np.load(os.path.join(FEATURES_DIR, file), allow_pickle=True)
        descriptors = data["descriptors"]

        # Limit descriptors to the first MAX_DB_DESCRIPTORS
        if descriptors.shape[0] > MAX_DB_DESCRIPTORS:
            descriptors = descriptors[:MAX_DB_DESCRIPTORS]

        db_features[loc] = descriptors
        print(f"Loaded {loc}: {descriptors.shape[0]} descriptors")

print("\nDatabase loaded successfully!")
print(f"Total locations: {len(db_features)}\n")

# Run the FLASK App
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
        # Lowe's ratio test
        good_matches = [
            m_n[0] for m_n in matches
            if len(m_n) == 2 and m_n[0].distance < RATIO_THRESH * m_n[1].distance
        ]
        score = len(good_matches)

        if score > best_score:
            best_score = score
            best_loc = loc

    elapsed = time.time() - start_time
    return best_loc, best_score, elapsed

@app.route('/predict_location', methods=['POST'])
def predict_location():
    # Accept multiple images (image0, image1, image2) for your 3-photo app
    images = [f for f in request.files if f.startswith("image")]
    if not images:
        return jsonify({"error": "No image uploaded"}), 400

    best_location, best_score, total_elapsed = None, 0, 0.0

    try:
        # Process each uploaded image and pick the one with the highest good_matches
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
