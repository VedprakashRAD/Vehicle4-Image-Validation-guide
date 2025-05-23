# User Guide: Vehicle Image Validation System

This guide explains how to use the Vehicle Image Validation System to upload and validate vehicle images from different angles (front, rear, left, right).

## Table of Contents

1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Using the API Directly](#using-the-api-directly)
4. [Using the YOLOv8 Preprocessor](#using-the-yolov8-preprocessor)
5. [Image Capture Guidelines](#image-capture-guidelines)
6. [Understanding Validation Results](#understanding-validation-results)
7. [Training the Model](#training-the-model)
8. [Troubleshooting](#troubleshooting)

## System Overview

The Vehicle Image Validation System allows you to:

- Upload images of vehicles from four angles (front, rear, left, right)
- Validate image quality (blur, lighting, etc.)
- Verify that images show the correct vehicle perspective
- Detect if the full vehicle is visible in the frame
- Classify vehicle sides using AI
- Save validated images as a dataset
- Train the AI model with validated images

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`)
- Internet connection for downloading YOLOv8 model (first time only)

### Starting the API Server

1. Activate your Python virtual environment:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Start the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 9000 --reload
   ```

3. Verify the server is running by opening http://localhost:9000/ in your browser
   - You should see a welcome message
   - API documentation is available at http://localhost:9000/docs

## Using the API Directly

### Step 1: Create a Session

Before uploading images, you need to create a session:

```bash
curl -X POST http://localhost:9000/session
```

This will return a session ID that you'll use for all subsequent requests:

```json
{
  "session_id": "ee757e95-896d-4a00-a4c6-3f19155bf939",
  "created_at": "2023-09-21T14:35:22.123456",
  "expires_in_hours": 24,
  "message": "Session created successfully"
}
```

### Step 2: Upload Images

Upload images one at a time, specifying which side of the vehicle is shown:

```bash
curl -X POST http://localhost:9000/upload/YOUR_SESSION_ID \
  -F "file=@/path/to/car_front.jpg" \
  -F "claimed_side=front"
```

Replace `YOUR_SESSION_ID` with the session ID you received in Step 1.

Repeat this process for all four sides:
- front
- rear
- left
- right

Alternatively, you can use the bulk upload endpoint to upload multiple images at once:

```bash
curl -X POST http://localhost:9000/upload-bulk/YOUR_SESSION_ID \
  -F "front_image=@/path/to/car_front.jpg" \
  -F "rear_image=@/path/to/car_rear.jpg" \
  -F "left_image=@/path/to/car_left.jpg" \
  -F "right_image=@/path/to/car_right.jpg"
```

### Step 3: Check Session Status

To check which sides you've successfully uploaded:

```bash
curl -X GET http://localhost:9000/status/YOUR_SESSION_ID
```

The response will show you which sides are uploaded and which are still missing:

```json
{
  "session_id": "ee757e95-896d-4a00-a4c6-3f19155bf939",
  "created_at": "2023-09-21T14:35:22.123456",
  "uploaded_sides": ["front", "left"],
  "upload_info": {
    "front": {
      "uploaded_at": "2023-09-21T14:36:15.654321",
      "validation_passed": true
    },
    "left": {
      "uploaded_at": "2023-09-21T14:40:22.123456",
      "validation_passed": true
    }
  },
  "missing_sides": ["rear", "right"],
  "is_complete": false
}
```

### Step 4: Download Images (Optional)

Once you've uploaded images, you can download them as a ZIP file:

```bash
curl -X GET http://localhost:9000/download/YOUR_SESSION_ID --output vehicle_images.zip
```

### Step 5: Save as Training Data (Optional)

If you want to contribute your images to improve the AI model:

```bash
curl -X POST http://localhost:9000/save-as-training/YOUR_SESSION_ID
```

Note: This requires all four sides to be uploaded successfully.

## Using the YOLOv8 Preprocessor

The YOLOv8 preprocessor helps detect vehicles in images and crop them properly. This is especially useful if:
- Your original images contain multiple vehicles
- The vehicle doesn't fill most of the frame
- You want to standardize your images before uploading

### Processing a Single Image

```bash
python run_yolo.py --input path/to/image.jpg --output output_folder --save-crops
```

### Processing Multiple Images

```bash
python run_yolo.py --input path/to/images_directory --output output_folder --save-crops --save-results
```

### Options

- `--model-size`: YOLOv8 model size: n(ano), s(mall), m(edium), l(arge), x(large) [default: m]
- `--conf`: Confidence threshold [default: 0.25]
- `--save-crops`: Save cropped vehicle images
- `--save-results`: Save detection results as annotated images
- `--device`: Device to run inference on (cpu, cuda, 0, 1, etc.)

### Viewing Results

After running the script:
1. Check the `/output_folder/crops` directory for cropped vehicle images
2. Use these cropped images when uploading to the API
3. Look at `/output_folder/*_result.jpg` files to see detection visualizations (if `--save-results` was used)

## Image Capture Guidelines

To get the best validation results, follow these guidelines when capturing or selecting vehicle images:

### General Requirements
- **Resolution**: Minimum 800x600 pixels (larger is better)
- **Focus**: Image should be sharp and clear, not blurry
- **Lighting**: Good lighting conditions, not too dark or too bright
- **Distance**: Full vehicle should be visible with some margin around it
- **Angle**: Camera should be positioned appropriately for the claimed side

### For Front View
- Position yourself directly in front of the vehicle
- Ensure license plate is visible if applicable
- Both headlights should be clearly visible
- The entire front of the vehicle should be in frame

### For Rear View
- Position yourself directly behind the vehicle  
- Ensure license plate is visible if applicable
- Both taillights should be clearly visible
- The entire rear of the vehicle should be in frame

### For Side Views (Left/Right)
- Position yourself perpendicular to the side of the vehicle
- Capture the entire length of the vehicle
- All doors should be visible
- Allow some margin above and below the vehicle

## Understanding Validation Results

When you upload an image, the system performs several checks and returns detailed results. Here's how to interpret them:

### Validation Categories

1. **Dimensions**: Checks if the image meets minimum resolution requirements
2. **Blur**: Analyzes image sharpness
3. **Lighting**: Ensures proper exposure and contrast
4. **Vehicle Detection**: Verifies a vehicle is present in the image
5. **Perspective**: Checks if the image shows the claimed vehicle side
6. **Visibility**: Ensures the full vehicle is visible
7. **Classification**: Uses AI to classify which side of the vehicle is shown

### Example Result Analysis

```json
"validation_results": {
  "dimensions": { 
    "is_acceptable": true, 
    "width": 1920, 
    "height": 1080,
    "message": "Image size: 1920x1080"
  },
  "blur": { 
    "is_acceptable": false, 
    "message": "Image appears to be blurry (score: 45.23, threshold: 100.00)."
  }
}
```

In this example:
- The image dimensions are acceptable (1920x1080)
- The image failed the blur check (score 45.23 is below the threshold of 100.00)

If any validation check fails, you should capture a new image that addresses the issue.

## Training the Model

The system allows you to improve the AI model by contributing validated images:

1. Upload and validate images for all four sides of a vehicle
2. Save the images as training data using the `/save-as-training/` endpoint
3. Once enough training data is collected, train the model using the `/train-model` endpoint

Note: Training requires at least 5 images per side.

## Troubleshooting

### API Connection Issues

- **Server not responding**: Ensure the FastAPI server is running (`uvicorn main:app --host 0.0.0.0 --port 9000`)
- **Connection refused**: Check that the port (9000) is not blocked by a firewall
- **Slow uploads**: Large images may take time to process; consider resizing them

### Image Validation Failures

- **Blur detection failure**: Ensure good lighting and hold the camera steady
- **"No vehicle detected"**: Make sure the vehicle fills a substantial portion of the frame
- **Perspective errors**: Double-check that you're capturing the correct side
- **"Vehicle appears to be cut off"**: Move further back to capture the entire vehicle

### YOLOv8 Issues

- **Model download errors**: Ensure internet connection is active
- **CUDA errors**: If using GPU, ensure CUDA is properly installed
- **No detections**: Try decreasing the confidence threshold (`--conf 0.1`)

### Other Common Issues

- **Session expired**: Sessions expire after 24 hours (configurable); create a new session
- **"An image for this side was already uploaded"**: Delete the session and create a new one, or use a different side
- **Out of memory**: If using large images, consider processing fewer at a time

If problems persist, check the server logs for more detailed error information. 