import os
import cv2
import numpy as np
import time
from flask import Flask, request, jsonify

# Configuration
FEATURES_DIR = r"C:/Users/mhmda/Downloads/FYP/dataset_opencv/features_npz"
MAX_DB_DESCRIPTORS = 15000   # limit per location for faster matching
RATIO_THRESH = 0.75           # Lowe's ratio test threshold

# ORB detector and BFMatcher
orb = cv2.ORB_create(nfeatures=5000)
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

# Initialize Flask app
app = Flask(__name__)

def recognize_location_from_image(img):
    """Extract ORB features and compare with database to find best match."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des_query = orb.detectAndCompute(gray, None)
    
    if des_query is None or len(des_query) == 0:
        return None, 0

    best_loc, best_score = None, 0

    for loc, des_db in db_features.items():
        matches = bf.knnMatch(des_query, des_db, k=2)

        # Apply Lowe's ratio test
        good_matches = [m_n[0] for m_n in matches if len(m_n) == 2 and m_n[0].distance < RATIO_THRESH * m_n[1].distance]
        score = len(good_matches)

        if score > best_score:
            best_score = score
            best_loc = loc

    return best_loc, best_score


@app.route('/predict_location', methods=['POST'])
def predict_location():
    # Handle up to 3 uploaded images and return the best matching location.
    start_time = time.time()
    images = []

    # Collect all uploaded images
    for i in range(3):
        file = request.files.get(f'image{i}')
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)

    if len(images) == 0:
        return jsonify({"error": "No valid images uploaded"}), 400

    # Aggregate scores from all images
    total_scores = {}
    for img in images:
        loc, score = recognize_location_from_image(img)
        if loc:
            total_scores[loc] = total_scores.get(loc, 0) + score

    if not total_scores:
        response = {"predicted_location": "Unknown", "good_matches": 0}
    else:
        best_loc = max(total_scores, key=total_scores.get)
        response = {
            "predicted_location": best_loc,
            "good_matches": int(total_scores[best_loc]),
        }

    response["elapsed_time_sec"] = round(time.time() - start_time, 2)
    print(f"Prediction: {response}")
    return jsonify(response)


if __name__ == '__main__':
    print("Starting server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)
