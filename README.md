# aynpath-server

## Overview

This repository contains the backend service for indoor localization and navigation by providing a Flask API named (/predict_location) that receives an image from the user, extracts ORB features, and matches them against a precomputed database of descriptors to predict the user’s current location. Returns the predicted location, number of good matches, and processing time.

## Features

* ORB-based image feature matching  
* Location classification (classical ML or CNN)  
* Flask REST API for communication with Ayn-Path mobile app  

## Requirements
- Python 3.8+  
- Flask  
- OpenCV  
- NumPy


## Usage
(1) Start the server
```python
python server.py
```

(2) By default, the API runs at:
```python
http://127.0.0.1:5000/
```

(3) Example Request
```bash
curl -X POST http://127.0.0.1:5000/predict_location \
  -F "image=@sample.jpg"
```

(3)Example Response
```json
{
  "predicted_location": "hallway_cafe",
  "good_matches": 120,
  "elapsed_time_sec": 0.85
}
```
## Limitations
* Works only with Ayn-Path dataset (IIUM environment)
* Accuracy depends on lighting and camera angle
* Processing speed varies with device performance
