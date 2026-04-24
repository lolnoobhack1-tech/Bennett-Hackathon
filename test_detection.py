#!/usr/bin/env python3
"""Test YOLO detection directly to debug the issue."""

import cv2
import numpy as np
from ultralytics import YOLO

def test_yolo_detection():
    # Load the YOLO model
    model = YOLO("yolov8n.pt")
    
    # Test with a simple test image
    print("Testing YOLO model with sample image...")
    
    # Create a test image or load one if available
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)  # Gray background
    
    # Run detection
    results = model.predict(test_image, conf=0.15, verbose=True)
    
    print(f"Number of results: {len(results)}")
    if results and results[0].boxes is not None:
        print(f"Number of detections: {len(results[0].boxes)}")
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0].cpu().item())
            conf = float(box.conf[0].cpu().item())
            print(f"Detection {i}: Class {cls_id}, Confidence {conf:.3f}")
    else:
        print("No detections found")
    
    # Test with actual image if provided
    try:
        # Try to load a test image
        img_path = "test1234.jpg"  # Change this to your actual test image path
        test_img = cv2.imread(img_path)
        if test_img is not None:
            print(f"\nTesting with actual image: {img_path}")
            results = model.predict(test_img, conf=0.15, verbose=True)
            print(f"Number of results: {len(results)}")
            if results and results[0].boxes is not None:
                print(f"Number of detections: {len(results[0].boxes)}")
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0].cpu().item())
                    conf = float(box.conf[0].cpu().item())
                    print(f"Detection {i}: Class {cls_id}, Confidence {conf:.3f}")
            else:
                print("No detections found in actual image")
        else:
            print(f"Could not load image: {img_path}")
    except Exception as e:
        print(f"Error testing with actual image: {e}")

if __name__ == "__main__":
    test_yolo_detection()
