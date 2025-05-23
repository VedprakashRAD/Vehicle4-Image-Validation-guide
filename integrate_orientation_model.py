import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import io
from PIL import Image

# Define constants
MODEL_PATH = "orientation_model_efficientnet_20250523-172916/best_model.h5"
CLASS_NAMES = [
    "front_view",
    "left_side_view",
    "rear_view",
    "right_side_view"
]

# Mapping from class names to user-friendly names
CLASS_MAPPING = {
    "front_view": "Front",
    "rear_view": "Rear",
    "left_side_view": "Left",
    "right_side_view": "Right"
}

# Load the model (lazy loading)
orientation_model = None

def load_orientation_model():
    """Load the orientation model if not already loaded"""
    global orientation_model
    if orientation_model is None:
        try:
            orientation_model = load_model(MODEL_PATH)
            print(f"Orientation model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading orientation model: {e}")
            return None
    return orientation_model

def preprocess_image(image_data, target_size=(224, 224)):
    """Preprocess an image for prediction"""
    # Convert to numpy array
    if isinstance(image_data, bytes):
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        # Assume it's already a numpy array
        img = image_data
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def detect_vehicle_orientation(image_data):
    """Detect the orientation of a vehicle in an image"""
    # Load the model
    model = load_orientation_model()
    if model is None:
        raise ValueError("Failed to load orientation model")
    
    try:
        # Preprocess the image
        img = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(img)[0]
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Lower the confidence threshold for better usability with our model
        if confidence < 0.35:
            return {
                "has_vehicle": False,
                "message": "No vehicle is detected",
                "orientation": None,
                "confidence": confidence
            }
        
        # Map class name to user-friendly name
        orientation = CLASS_MAPPING.get(predicted_class, predicted_class)
        
        return {
            "has_vehicle": True,
            "orientation": orientation,
            "confidence": confidence,
            "class_name": predicted_class
        }
    
    except Exception as e:
        raise ValueError(f"Error detecting vehicle orientation: {e}")

def verify_orientation(image_data, expected_orientation):
    """Verify if the image has the expected vehicle orientation"""
    # Detect orientation
    result = detect_vehicle_orientation(image_data)
    
    # If no vehicle is detected
    if not result["has_vehicle"]:
        return {
            "is_valid": False,
            "message": "No vehicle is detected",
            "confidence": result.get("confidence", 0.0)
        }
    
    # Get detected orientation
    detected_orientation = result["orientation"]
    
    # Check if orientation matches expected
    if detected_orientation.lower() == expected_orientation.lower():
        return {
            "is_valid": True,
            "message": f"{detected_orientation} side uploaded successfully",
            "confidence": result["confidence"]
        }
    else:
        return {
            "is_valid": False,
            "message": f"Mismatch: {detected_orientation} side is detected for the intended {expected_orientation} side",
            "confidence": result["confidence"]
        }

# Example FastAPI integration
app = FastAPI()

# Add CORS middleware to allow requests from the mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class OrientationResponse(BaseModel):
    is_valid: bool
    message: str
    confidence: Optional[float] = None

@app.post("/verify-orientation/{expected_orientation}", response_model=OrientationResponse)
async def verify_vehicle_orientation(
    expected_orientation: str,
    file: UploadFile = File(...)
):
    """Verify if the uploaded image has the expected vehicle orientation"""
    # Validate expected orientation
    valid_orientations = ["front", "rear", "left", "right"]
    if expected_orientation.lower() not in valid_orientations:
        raise HTTPException(status_code=400, detail=f"Invalid orientation. Must be one of: {', '.join(valid_orientations)}")
    
    # Read image file
    image_data = await file.read()
    
    try:
        # Verify orientation
        result = verify_orientation(image_data, expected_orientation)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "ok", "model": os.path.exists(MODEL_PATH)}

if __name__ == "__main__":
    # If the model file exists, run the FastAPI server
    if os.path.exists(MODEL_PATH):
        import uvicorn
        print(f"Starting server on port 9090. Model path: {MODEL_PATH}")
        uvicorn.run(app, host="0.0.0.0", port=9090)
    else:
        print(f"Model file not found at {MODEL_PATH}. Train the model first.")
        print("You can run the training pipeline with: ./run_pipeline.sh") 