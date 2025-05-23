#!/usr/bin/env python3
"""
YOLOv8 Vehicle Detection Script

This script uses YOLOv8 to detect vehicles in images and save the bounding box coordinates.
The YOLO model will detect vehicles, crop images to the vehicle area, and save both
the cropped images and the detection results.

Install required packages: pip install ultralytics opencv-python pillow
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLOv8 vehicle detection script")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model-size", type=str, default="m", choices=["n", "s", "m", "l", "x"], 
                        help="YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped vehicle images")
    parser.add_argument("--save-results", action="store_true", help="Save detection results as images")
    parser.add_argument("--device", type=str, default="", help="Device to run inference on (cpu, cuda, 0, 1, etc.)")
    
    return parser.parse_args()

def process_image(model, image_path, output_dir, conf, save_crops=False, save_results=False):
    """Process a single image with YOLOv8"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    crops_dir = Path(output_dir) / "crops"
    if save_crops:
        crops_dir.mkdir(exist_ok=True)
    
    # Load and process the image
    try:
        # Run inference
        results = model(image_path, conf=conf, classes=[2, 3, 5, 7])  # Class IDs for vehicle-related categories
        
        # If no vehicles found, return empty list
        if len(results) == 0 or len(results[0].boxes) == 0:
            print(f"No vehicles detected in {image_path}")
            return []
        
        # Process results
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error reading image: {image_path}")
            return []
            
        img_height, img_width = img.shape[:2]
        detections = []
        
        # Process each detection
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            # Get class name
            class_name = results[0].names[class_id]
            
            # Only keep vehicle classes
            if class_name not in ['car', 'bus', 'truck', 'motorcycle']:
                continue
                
            # Calculate area and ensure minimum size
            area = (x2 - x1) * (y2 - y1)
            if area < 5000:  # Skip very small detections
                continue
                
            # Add padding (5% on each side)
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            
            # Ensure padding doesn't go outside image boundaries
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(img_width, x2 + pad_x)
            y2_pad = min(img_height, y2 + pad_y)
            
            # Save cropped image if requested
            if save_crops:
                crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
                crop_filename = f"{Path(image_path).stem}_crop_{i}.jpg"
                cv2.imwrite(str(crops_dir / crop_filename), crop)
            
            # Add detection to results
            detection = {
                "image": str(image_path),
                "crop_filename": f"{Path(image_path).stem}_crop_{i}.jpg" if save_crops else None,
                "class": class_name,
                "confidence": confidence,
                "bbox": {
                    "x1": int(x1_pad),
                    "y1": int(y1_pad),
                    "x2": int(x2_pad),
                    "y2": int(y2_pad),
                    "width": int(x2_pad - x1_pad),
                    "height": int(y2_pad - y1_pad)
                }
            }
            detections.append(detection)
            
            # Draw bounding box on the image for visualization
            if save_results:
                color = (0, 255, 0)  # Green bounding box
                cv2.rectangle(img, (x1_pad, y1_pad), (x2_pad, y2_pad), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(img, label, (x1_pad, y1_pad - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image if requested
        if save_results and len(detections) > 0:
            result_path = Path(output_dir) / f"{Path(image_path).stem}_result.jpg"
            cv2.imwrite(str(result_path), img)
            
        # Save JSON results
        if len(detections) > 0:
            json_path = Path(output_dir) / f"{Path(image_path).stem}_detections.json"
            with open(json_path, 'w') as f:
                json.dump(detections, f, indent=2)
                
        return detections
                
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

def process_directory(model, input_dir, output_dir, conf, save_crops=False, save_results=False):
    """Process all images in a directory"""
    input_path = Path(input_dir)
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process images
    all_detections = []
    for i, image_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_file}")
        detections = process_image(model, image_file, output_dir, conf, save_crops, save_results)
        all_detections.extend(detections)
        
    # Save overall results
    if all_detections:
        summary_path = Path(output_dir) / "all_detections.json"
        with open(summary_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
            
    print(f"Processed {len(image_files)} images, detected {len(all_detections)} vehicles")
    return all_detections

def main():
    """Main function"""
    args = parse_args()
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8{args.model_size} model...")
    model = YOLO(f"yolov8{args.model_size}.pt")
    
    # Set device if specified
    if args.device:
        model.to(args.device)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Check if input is a directory or a single image
    input_path = Path(args.input)
    if input_path.is_dir():
        print(f"Processing directory: {args.input}")
        process_directory(model, args.input, args.output, args.conf, args.save_crops, args.save_results)
    else:
        print(f"Processing image: {args.input}")
        process_image(model, args.input, args.output, args.conf, args.save_crops, args.save_results)
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 