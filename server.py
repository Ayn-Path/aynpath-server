# server.py
import os
import cv2
import numpy as np
import time
from flask import Flask, request, jsonify

# Configuration
FEATURES_DIR = r"C:/Users/mhmda/Downloads/dataset_opencv/features_npz"
MAX_DB_DESCRIPTORS = 100000  # limit per location for faster matching
RATIO_THRESH = 0.75         # Lowe's ratio test threshold

# ORB detector and BFMatcher
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Load Database
db_features = {}
for file in os.listdir(FEATURES_DIR):
    if file.endswith("_features.npz"):
        loc = file.replace("_features.npz", "")
        data = np.load(os.path.join(FEATURES_DIR, file))
        descriptors = data["descriptors"]

        # Sample descriptors if too many
        if descriptors.shape[0] > MAX_DB_DESCRIPTORS:
            idx = np.random.choice(descriptors.shape[0], MAX_DB_DESCRIPTORS, replace=False)
            descriptors = descriptors[idx]

        db_features[loc] = descriptors

print("Database loaded:")
for loc, des in db_features.items():
    print(f"{loc}: {des.shape[0]} descriptors")

# Run the FLASK App
app = Flask(__name__)

def recognize_location_from_image(img):
    """
    Input: OpenCV image
    Output: predicted location, good_matches count, elapsed_time
    """
    start_time = time.time()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des_query = orb.detectAndCompute(gray, None)
    
    if des_query is None or len(des_query) == 0:
        return None, 0, time.time() - start_time
    
    best_loc, best_score = None, 0

    for loc, des_db in db_features.items():
        matches = bf.knnMatch(des_query, des_db, k=2)
        
        # Lowe's ratio test
        good_matches = [m_n[0] for m_n in matches if len(m_n) == 2 and m_n[0].distance < RATIO_THRESH * m_n[1].distance]
        score = len(good_matches)

        if score > best_score:
            best_score = score
            best_loc = loc

    elapsed = time.time() - start_time
    return best_loc, best_score, elapsed

@app.route('/predict_location', methods=['POST'])
def predict_location():
    if 'image' not in request.files: # To make sure that the image retrieve from the camera's phone are in the database
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        location, score, elapsed = recognize_location_from_image(img)

        response = {
            "predicted_location": location or "Unknown",
            "good_matches": score,
            "elapsed_time_sec": round(elapsed, 2)
        }

        print(f"Prediction: {response}")  # Log for debugging
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)