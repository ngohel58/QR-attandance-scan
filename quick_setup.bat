@echo off
python generate_android_project.py %*
python verify_project.py
echo Project ready in .\DepthEstimationApp
