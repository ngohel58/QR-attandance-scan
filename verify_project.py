#!/usr/bin/env python3
"""Verify generated Android project for required files."""
from pathlib import Path
import sys

REQUIRED_PATHS = [
    'build.gradle',
    'settings.gradle',
    'app/build.gradle',
    'app/src/main/AndroidManifest.xml',
    'app/src/main/java',
    'app/src/main/python',
    'app/src/main/res/layout/activity_main.xml',
]

def main(project_dir='DepthEstimationApp'):
    base = Path(project_dir)
    missing = []
    for rel in REQUIRED_PATHS:
        if not (base / rel).exists():
            missing.append(rel)
    if missing:
        print('Missing files:')
        for m in missing:
            print(' -', m)
        return 1
    print('Project structure looks good.')
    return 0

if __name__ == '__main__':
    project = sys.argv[1] if len(sys.argv) > 1 else 'DepthEstimationApp'
    raise SystemExit(main(project))
