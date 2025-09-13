# App Server

Backend service for indoor localization and navigation by providing a Flask API named (/predict_location) that receives an image from the user, extracts ORB features, and matches them against a precomputed database of descriptors to predict the user’s current location. Returns the predicted location, number of good matches, and processing time.
