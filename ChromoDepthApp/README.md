# ChromoDepthApp

Prototype Android application demonstrating deployment of a depth-estimation ONNX model converted for Samsung Galaxy S24 using QAI Hub.

## Features

- Load image from gallery
- Generate depth map using ONNX Runtime 1.22.0
- Apply chromostereopsis effect with adjustable threshold
- Preview depth map and effect with pinch-to-zoom (PhotoView)
- Save generated images (TODO)

## Setup

1. Place optimized `model.onnx` and `model.data` under `app/src/main/assets/`.
2. Open project in Android Studio (Giraffe or newer).
3. Build and run on a device supporting ONNX Runtime (tested target: Samsung Galaxy S24, Android 14).

## Notes

This project provides a skeleton implementation. Additional polish, error handling, and UI enhancements may be added as needed.
