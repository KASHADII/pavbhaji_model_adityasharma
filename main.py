"""
Root Main Runner for DriveBuddyAI Pav Bhaji Text Classification Project.

Delegates execution to the drivebuddy_pavbhaji_classifier pipeline.
"""

import os
import sys

# Add drivebuddy_pavbhaji_classifier to sys.path
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drivebuddy_pavbhaji_classifier")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from main import main

if __name__ == "__main__":
    main()
