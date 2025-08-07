#!/bin/bash
set -e

# Quick setup for generating Android project
pip3 install --user Pillow numpy >/dev/null 2>&1 || true
python3 generate_android_project.py "$@"
python3 verify_project.py

cat <<MSG
Project ready in ./DepthEstimationApp
Open in Android Studio and set sdk.dir in local.properties
MSG
