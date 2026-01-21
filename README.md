# aynpath-server

## Overview
This repository contains the ORB (Oriented FAST and Rotated BRIEF) feature processing server developed for the Final Year Project (FYP) **AynPath**. The server supports image-based indoor localization by extracting and matching ORB features from input images. The processed results are used by the AynPath application to assist Augmented Reality (AR) navigation in indoor environments where GPS is unavailable or unreliable.

## Purpose
The purpose of this server is to provide a backend component for the visual feature processing. The server:
- Receives image input from the client application
- Extracts ORB keypoints and descriptors
- Performs feature matching
- Returns matching results to support localization and navigation

ORB is chosen due to its computational efficiency, rotation invariance, and suitability for real-time indoor applications.

## Implementation
The server is implemented in Python using the OpenCV library. A lightweight HTTP server framework is used to expose endpoints that accept image data and return ORB feature matching results.

The main processing flow is as follows:
1. Receive image data from the client
2. Convert the image to grayscale
3. Extract ORB keypoints and descriptors
4. Perform feature matching
5. Return matching results to the client

## Technologies Used
- Python 3
- OpenCV (ORB feature detector)
- Flask (HTTP server)
- REST-based communication

## Repository Structure
- server.py # Main ORB feature processing server
- server2.py # Alternative / experimental implementation
- requirements.txt # Python dependencies
- README.md # Project documentation

## Installation
1. Clone the repository:
```bash
   git clone https://github.com/Ayn-Path/aynpath-server.git
   cd aynpath-server
```
2. Install the required dependencies:
```bash
   pip install -r requirements.txt
```
## Running the Server
After installing the dependencies, start the server using:
```bash
python server.py
```
The server will run locally and listen for incoming image processing requests from the AynPath application.

## Notes
- This server is developed for academic and experimental purposes as part of a Final Year Project.
- Performance may vary depending on hardware capability and image resolution.
- Further optimization and feature expansion can be explored in future work.
