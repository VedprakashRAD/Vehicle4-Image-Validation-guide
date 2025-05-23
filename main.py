from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List, Optional, Dict, Any
import os
import uuid
import shutil
import io
import zipfile
from PIL import Image
import imagehash
import numpy as np
from pathlib import Path
import uvicorn
import time
import json
from datetime import datetime
import logging
import torch
import easyocr

# Import our custom modules
from ml_model import VehicleSideClassifier, prepare_dataset
from image_utils import (
    calculate_image_hash, 
    is_duplicate, 
    comprehensive_image_check,
    check_vehicle_perspective
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create app instance
app = FastAPI(
    title="Vehicle Image Validation API",
    description="API for validating and managing vehicle images from different angles",
    version="1.0.0"
)

# Add CORS middleware to allow requests from Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your Flutter app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TRAINING_DIR = Path("training_data")
TRAINING_DIR.mkdir(exist_ok=True)

# Load configuration
CONFIG_FILE = Path("config.json")
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    config = {
        "min_width": 800,
        "min_height": 600,
        "model_path": str(MODEL_DIR / "vehicle_side_model.h5"),
        "detection_confidence_threshold": 0.5,
        "duplicate_threshold": 5,
        "session_expiry_hours": 24
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

# In-memory storage for session data (replace with database in production)
sessions = {}

# Initialize the vehicle side classifier
try:
    classifier = VehicleSideClassifier(config.get("model_path"))
except Exception as e:
    logger.error(f"Error loading classifier model: {e}")
    classifier = VehicleSideClassifier()  # Fallback to new model

# Load YOLOv8 model for vehicle and number plate detection
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    logger.error(f"Error loading YOLOv5 model: {e}")
    yolo_model = None

# Load EasyOCR reader for number plate extraction
try:
    ocr_reader = easyocr.Reader(['en'])
except Exception as e:
    logger.error(f"Error loading EasyOCR: {e}")
    ocr_reader = None

# Add number plate storage to session
for s in sessions.values():
    if "number_plate" not in s:
        s["number_plate"] = None
    if "towed_number_plate" not in s:
        s["towed_number_plate"] = None
    if "towing_number_plate" not in s:
        s["towing_number_plate"] = None

def get_session_path(session_id: str) -> Path:
    """Get the path for a session's uploads"""
    session_path = UPLOAD_DIR / session_id
    session_path.mkdir(exist_ok=True)
    return session_path

def create_session():
    """Create a new session for tracking uploads"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "front": None,
        "rear": None,
        "left": None, 
        "right": None,
        "hashes": []  # Store image hashes for duplicate detection
    }
    return session_id

def is_valid_session(session_id: str) -> bool:
    """Check if a session exists and is valid"""
    if session_id not in sessions:
        return False
    
    # Check session expiry (optional)
    if "created_at" in sessions[session_id]:
        created_at = datetime.fromisoformat(sessions[session_id]["created_at"])
        expiry_hours = config.get("session_expiry_hours", 24)
        if (datetime.now() - created_at).total_seconds() > expiry_hours * 3600:
            # Clean up expired session
            if session_id in sessions:
                del sessions[session_id]
            return False
    
    return True

@app.get("/")
async def root():
    """
    Root endpoint that serves the download page
    """
    return FileResponse("download.html", media_type="text/html")

@app.post("/session")
def create_new_session():
    """Create a new upload session"""
    session_id = create_session()
    session_path = get_session_path(session_id)
    
    return {
        "session_id": session_id,
        "created_at": sessions[session_id]["created_at"],
        "expires_in_hours": config.get("session_expiry_hours", 24),
        "message": "Session created successfully"
    }

@app.post("/upload/{session_id}")
async def upload_image(
    session_id: str,
    file: UploadFile = File(...),
    claimed_side: str = Form(...),
):
    """
    Upload and validate a vehicle image
    
    - **session_id**: ID of the upload session
    - **file**: Image file to upload
    - **claimed_side**: Side of the vehicle shown ('front', 'rear', 'left', 'right')
    
    Returns a detailed validation result and status
    """
    # Check if session exists and is valid
    if not is_valid_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    # Validate claimed side
    if claimed_side not in ["front", "rear", "left", "right"]:
        raise HTTPException(status_code=400, detail="Invalid side. Must be front, rear, left, or right")
    
    # Check if this side was already uploaded
    if sessions[session_id][claimed_side] is not None:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "message": f"An image for the {claimed_side} side was already uploaded",
                "error_code": "DUPLICATE_SIDE"
            }
        )
    
    # Save the uploaded file temporarily
    session_path = get_session_path(session_id)
    temp_file_path = session_path / f"temp_{uuid.uuid4()}.jpg"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image
    try:
        # Open image for processing
        img = Image.open(temp_file_path)
        
        # --- YOLOv8 vehicle detection ---
        vehicle_detected = False
        if yolo_model is not None:
            results = yolo_model(str(temp_file_path))
            labels = results.pandas().xyxy[0]['name'].tolist()
            if any(label in ['car', 'motorcycle', 'bus', 'truck'] for label in labels):
                vehicle_detected = True
        else:
            logger.warning("YOLOv5 model not loaded, skipping vehicle detection.")
            vehicle_detected = True  # fallback for dev
        if not vehicle_detected:
            temp_file_path.unlink(missing_ok=True)
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No vehicle is detected.",
                    "error_code": "NO_VEHICLE"
                }
            )
        # --- End YOLOv8 vehicle detection ---
        
        # Calculate image hash for duplicate detection
        img_hash = calculate_image_hash(img)
        
        # Check for duplicates within this session
        is_dup, matching_hash = is_duplicate(
            img, 
            sessions[session_id]["hashes"], 
            threshold=config.get("duplicate_threshold", 5)
        )
        
        if is_dup:
            # Find which side the duplicate is from
            duplicate_side = None
            for side, info in sessions[session_id].items():
                if isinstance(info, dict) and info.get("hash") == matching_hash:
                    duplicate_side = side
                    break
            
            os.remove(temp_file_path)
            return JSONResponse(
                status_code=409,
                content={
                    "status": "error",
                    "message": f"This appears to be a duplicate of the {duplicate_side} image",
                    "error_code": "DUPLICATE_IMAGE"
                }
            )
        
        # Run comprehensive image checks
        is_acceptable, check_results = comprehensive_image_check(
            img, 
            claimed_side,
            min_width=config.get("min_width", 800),
            min_height=config.get("min_height", 600)
        )
        
        # Use classifier to classify the side of the vehicle
        predicted_side, confidence = classifier.predict(img)
        
        # Stricter validation for side classification
        side_match = predicted_side == claimed_side
        min_confidence = 0.7  # Increased from default
        has_sufficient_confidence = confidence >= min_confidence
        
        # Record detection results
        check_results["classification"] = {
            "claimed_side": claimed_side,
            "detected_side": predicted_side,
            "confidence": float(confidence),
            "min_confidence": float(min_confidence),
            "is_acceptable": side_match and has_sufficient_confidence
        }
        
        # Update overall acceptability - now stricter with side classification
        is_acceptable = is_acceptable and check_results["classification"]["is_acceptable"]
        
        # If the image passed all checks
        if is_acceptable:
            # Store the image
            final_path = session_path / f"{claimed_side}.jpg"
            os.rename(temp_file_path, final_path)
            
            # Update session data
            sessions[session_id][claimed_side] = {
                "path": str(final_path),
                "hash": img_hash,
                "upload_time": datetime.now().isoformat(),
                "check_results": check_results
            }
            sessions[session_id]["hashes"].append(img_hash)
            
            # Check completion status
            missing_sides = [side for side, info in sessions[session_id].items() 
                            if side in ["front", "rear", "left", "right"] and info is None]
            
            return {
                "status": "success",
                "message": f"{claimed_side.capitalize()} side uploaded successfully.",
                "predicted_side": predicted_side,
                "confidence": float(confidence),
                "validation_results": check_results,
                "missing_sides": missing_sides,
                "is_complete": len(missing_sides) == 0
            }
        else:
            # Failed validation
            os.remove(temp_file_path)
            
            # Determine the most important failure reason
            failure_reasons = []
            for check, result in check_results.items():
                if isinstance(result, dict) and not result.get("is_acceptable", True):
                    failure_reasons.append(result.get("message", f"Failed {check} check"))
            
            if not check_results["classification"]["is_acceptable"]:
                if side_match and not has_sufficient_confidence:
                    failure_message = f"Unable to confirm this is a {claimed_side} view with sufficient confidence"
                else:
                    failure_message = f"This appears to be a {predicted_side} view, not a {claimed_side} view"
            elif failure_reasons:
                failure_message = failure_reasons[0]
            else:
                failure_message = "Image validation failed"
            
            return JSONResponse(
                status_code=422,  # Unprocessable Entity
                content={
                    "status": "error",
                    "message": failure_message,
                    "validation_results": check_results,
                    "error_code": "VALIDATION_FAILED"
                }
            )
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        # Clean up on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@app.post("/upload-bulk/{session_id}")
async def upload_bulk_images(
    session_id: str,
    front_image: Optional[UploadFile] = File(None),
    rear_image: Optional[UploadFile] = File(None),
    left_image: Optional[UploadFile] = File(None),
    right_image: Optional[UploadFile] = File(None)
):
    """
    Upload multiple vehicle images at once
    
    - **session_id**: ID of the upload session
    - **front_image**: Front view image file (optional)
    - **rear_image**: Rear view image file (optional)
    - **left_image**: Left side view image file (optional) 
    - **right_image**: Right side view image file (optional)
    
    Returns upload status for each image
    """
    # Check if session exists
    if not is_valid_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    results = {}
    uploads = {
        "front": front_image,
        "rear": rear_image,
        "left": left_image,
        "right": right_image
    }
    
    # Process each uploaded image
    for side, file in uploads.items():
        if file is not None:
            try:
                # Check if this side was already uploaded
                if sessions[session_id][side] is not None:
                    results[side] = {
                        "status": "error",
                        "message": f"An image for the {side} side was already uploaded",
                        "error_code": "DUPLICATE_SIDE"
                    }
                    continue
                
                # Save the uploaded file temporarily
                session_path = get_session_path(session_id)
                temp_file_path = session_path / f"temp_{side}_{uuid.uuid4()}.jpg"
                
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Open image for processing
                img = Image.open(temp_file_path)
                
                # Calculate image hash for duplicate detection
                img_hash = calculate_image_hash(img)
                
                # Check for duplicates within this session
                is_dup, matching_hash = is_duplicate(
                    img, 
                    sessions[session_id]["hashes"], 
                    threshold=config.get("duplicate_threshold", 5)
                )
                
                if is_dup:
                    # Find which side the duplicate is from
                    duplicate_side = None
                    for s, info in sessions[session_id].items():
                        if isinstance(info, dict) and info.get("hash") == matching_hash:
                            duplicate_side = s
                            break
                    
                    os.remove(temp_file_path)
                    results[side] = {
                        "status": "error",
                        "message": f"This appears to be a duplicate of the {duplicate_side} image",
                        "error_code": "DUPLICATE_IMAGE"
                    }
                    continue
                
                # Run comprehensive image checks
                is_acceptable, check_results = comprehensive_image_check(
                    img, 
                    side,
                    min_width=config.get("min_width", 800),
                    min_height=config.get("min_height", 600)
                )
                
                # Use classifier to verify the side
                detected_side, confidence = classifier.predict(img)
                
                # Stricter validation for side classification
                side_match = detected_side == side
                min_confidence = 0.7  # Increased from default
                has_sufficient_confidence = confidence >= min_confidence
                
                # Record detection results
                check_results["classification"] = {
                    "claimed_side": side,
                    "detected_side": detected_side,
                    "confidence": float(confidence),
                    "min_confidence": float(min_confidence),
                    "is_acceptable": side_match and has_sufficient_confidence
                }
                
                # Update overall acceptability
                is_acceptable = is_acceptable and check_results["classification"]["is_acceptable"]
                
                # If the image passed all checks
                if is_acceptable:
                    # Store the image
                    final_path = session_path / f"{side}.jpg"
                    os.rename(temp_file_path, final_path)
                    
                    # Update session data
                    sessions[session_id][side] = {
                        "path": str(final_path),
                        "hash": img_hash,
                        "upload_time": datetime.now().isoformat(),
                        "check_results": check_results
                    }
                    sessions[session_id]["hashes"].append(img_hash)
                    
                    results[side] = {
                        "status": "success",
                        "message": f"Successfully uploaded {side} image",
                        "detected_side": detected_side,
                        "confidence": float(confidence)
                    }
                else:
                    # Failed validation
                    os.remove(temp_file_path)
                    
                    # Determine the most important failure reason
                    failure_reasons = []
                    for check, result in check_results.items():
                        if isinstance(result, dict) and not result.get("is_acceptable", True):
                            failure_reasons.append(result.get("message", f"Failed {check} check"))
                    
                    if not check_results["classification"]["is_acceptable"]:
                        if side_match and not has_sufficient_confidence:
                            failure_message = f"Unable to confirm this is a {side} view with sufficient confidence"
                        else:
                            failure_message = f"This appears to be a {detected_side} view, not a {side} view"
                    elif failure_reasons:
                        failure_message = failure_reasons[0]
                    else:
                        failure_message = "Image validation failed"
                    
                    results[side] = {
                        "status": "error",
                        "message": failure_message,
                        "error_code": "VALIDATION_FAILED"
                    }
                    
            except Exception as e:
                logger.error(f"Error processing {side} image: {e}")
                results[side] = {
                    "status": "error",
                    "message": f"Error processing image: {str(e)}",
                    "error_code": "PROCESSING_ERROR"
                }
                
                # Clean up on error
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    
    # Check completion status
    missing_sides = [side for side, info in sessions[session_id].items() 
                    if side in ["front", "rear", "left", "right"] and info is None]
    
    return {
        "results": results,
        "missing_sides": missing_sides,
        "is_complete": len(missing_sides) == 0
    }

@app.get("/status/{session_id}")
def check_status(session_id: str):
    """Check the status of uploaded images for a session"""
    if not is_valid_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session_data = sessions[session_id]
    
    # Get upload information
    upload_info = {}
    for side in ["front", "rear", "left", "right"]:
        if session_data[side] is not None:
            upload_info[side] = {
                "uploaded_at": session_data[side].get("upload_time", "unknown"),
                "validation_passed": True
            }
    
    uploaded_sides = list(upload_info.keys())
    missing_sides = [side for side in ["front", "rear", "left", "right"] if side not in uploaded_sides]
    
    return {
        "session_id": session_id,
        "created_at": session_data.get("created_at", "unknown"),
        "uploaded_sides": uploaded_sides,
        "upload_info": upload_info,
        "missing_sides": missing_sides,
        "is_complete": len(missing_sides) == 0
    }

@app.get("/download/{session_id}")
def download_images(session_id: str):
    """Download all images from a session as a ZIP file"""
    if not is_valid_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session_data = sessions[session_id]
    
    # Check if any images are uploaded
    uploaded_sides = [side for side in ["front", "rear", "left", "right"] if session_data[side] is not None]
    if not uploaded_sides:
        raise HTTPException(status_code=400, detail="No images have been uploaded in this session")
    
    # Create ZIP file in memory
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        for side in uploaded_sides:
            image_path = session_data[side]["path"]
            filename = os.path.basename(image_path)
            zip_file.write(image_path, arcname=f"{side}_{filename}")
    
    # Seek to the beginning of the stream
    zip_io.seek(0)
    
    # Return the ZIP file as a streaming response
    return StreamingResponse(
        zip_io, 
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=vehicle_images_{session_id}.zip"}
    )

@app.post("/save-as-training/{session_id}")
def save_as_training(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """Save the validated images as training data for the model"""
    if not is_valid_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session_data = sessions[session_id]
    
    # Check if all sides are uploaded
    missing_sides = [side for side in ["front", "rear", "left", "right"] if session_data[side] is None]
    if missing_sides:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Missing images for sides: {', '.join(missing_sides)}",
                "error_code": "INCOMPLETE_SESSION"
            }
        )
    
    # Create training directories
    for side in ["front", "rear", "left", "right"]:
        side_dir = TRAINING_DIR / side
        side_dir.mkdir(exist_ok=True)
    
    # Copy images to training directories
    for side in ["front", "rear", "left", "right"]:
        src_path = session_data[side]["path"]
        dst_path = TRAINING_DIR / side / f"{session_id}_{side}.jpg"
        shutil.copy2(src_path, dst_path)
    
    return {
        "status": "success",
        "message": "Images saved as training data successfully"
    }

@app.post("/train-model")
def train_model(background_tasks: BackgroundTasks):
    """Train the model using the collected training data"""
    # Check if enough training data exists
    min_samples_per_class = 5
    has_enough_data = True
    
    for side in ["front", "rear", "left", "right"]:
        side_dir = TRAINING_DIR / side
        side_dir.mkdir(exist_ok=True)
        
        files = list(side_dir.glob("*.jpg"))
        if len(files) < min_samples_per_class:
            has_enough_data = False
    
    if not has_enough_data:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Not enough training data. Need at least {min_samples_per_class} images per side.",
                "error_code": "INSUFFICIENT_DATA"
            }
        )
    
    # Add training task to background
    def train_in_background():
        global classifier
        
        try:
            # Prepare datasets
            train_data = prepare_dataset(str(TRAINING_DIR), batch_size=32)
            
            # Create validation split
            val_data = train_data.take(5)
            train_data = train_data.skip(5)
            
            # Train the model
            model_path = str(MODEL_DIR)
            classifier.train(train_data, val_data, epochs=20, save_path=model_path)
            
            # Update config
            config["model_path"] = str(MODEL_DIR / "final_model.h5")
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
                
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
    
    background_tasks.add_task(train_in_background)
    
    return {
        "status": "success",
        "message": "Model training started in the background"
    }

@app.get("/config")
def get_config():
    """Get the current configuration"""
    return config

@app.post("/config")
def update_config(updated_config: Dict[str, Any]):
    """Update the configuration"""
    global config
    
    # Update only allowed fields
    allowed_fields = [
        "min_width", 
        "min_height", 
        "detection_confidence_threshold",
        "duplicate_threshold",
        "session_expiry_hours"
    ]
    
    for field in allowed_fields:
        if field in updated_config:
            config[field] = updated_config[field]
    
    # Save updated config
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    
    return {
        "status": "success",
        "message": "Configuration updated successfully",
        "config": config
    }

@app.get("/download-app")
async def download_app():
    """Endpoint to download the Android APK file"""
    return FileResponse("app-debug.apk", filename="vehicle-image-validator.apk", media_type="application/vnd.android.package-archive")

@app.get("/sample-images/{side}")
async def get_sample_image(side: str):
    """
    Get a sample image showing the correct way to capture a specific vehicle side
    
    Args:
        side: Vehicle side (front, rear, left, right)
    
    Returns:
        Sample image file
    """
    if side not in ["front", "rear", "left", "right"]:
        raise HTTPException(status_code=400, detail="Invalid side. Must be front, rear, left, or right")
    
    # Define paths for sample images
    sample_images = {
        "front": Path("sample_images/front/sample_front.jpg"),
        "rear": Path("sample_images/rear/sample_rear.jpg"),
        "left": Path("sample_images/left/sample_left.jpg"),
        "right": Path("sample_images/right/sample_right.jpg"),
    }
    
    # Check if the sample image exists
    if not sample_images[side].exists():
        # If no sample image, return a placeholder
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": f"No sample image available for {side} view",
                "error_code": "SAMPLE_NOT_FOUND"
            }
        )
    
    return FileResponse(
        sample_images[side],
        filename=f"sample_{side}.jpg",
        media_type="image/jpeg"
    )

# Helper function to detect number plate and extract text
def detect_and_read_plate(image_path):
    if yolo_model is None or ocr_reader is None:
        return None, "Detection/OCR model not loaded"
    results = yolo_model(str(image_path))
    df = results.pandas().xyxy[0]
    # Find plate-like objects (YOLOv5s doesn't have 'license plate' class, so we use the largest box as a proxy)
    if len(df) == 0:
        return None, "No objects detected"
    # Use the largest detected object as the plate (for demo; in prod, use a plate-specific model)
    largest = df.iloc[(df['xmax']-df['xmin']).idxmax()]
    x1, y1, x2, y2 = map(int, [largest['xmin'], largest['ymin'], largest['xmax'], largest['ymax']])
    img = Image.open(image_path)
    plate_img = img.crop((x1, y1, x2, y2))
    plate_img_np = np.array(plate_img)
    result = ocr_reader.readtext(plate_img_np)
    if result:
        return result[0][1], None
    return None, "OCR failed"

@app.post("/upload-towed/{session_id}")
async def upload_towed_vehicle(
    session_id: str,
    file: UploadFile = File(...),
):
    """
    Upload an image of the towed (damaged) vehicle, extract and compare number plate.
    """
    if not is_valid_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    # Save file
    session_path = get_session_path(session_id)
    temp_file_path = session_path / f"towed_{uuid.uuid4()}.jpg"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Detect vehicle
    vehicle_detected = False
    if yolo_model is not None:
        results = yolo_model(str(temp_file_path))
        labels = results.pandas().xyxy[0]['name'].tolist()
        if any(label in ['car', 'motorcycle', 'bus', 'truck'] for label in labels):
            vehicle_detected = True
    else:
        vehicle_detected = True
    if not vehicle_detected:
        temp_file_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "No vehicle is detected in the towed image.",
                "error_code": "NO_VEHICLE"
            }
        )
    # Detect and read number plate
    plate_text, err = detect_and_read_plate(temp_file_path)
    if plate_text:
        sessions[session_id]["towed_number_plate"] = plate_text
        # Compare with original (use first available plate from front/rear)
        orig_plate = sessions[session_id].get("number_plate")
        if not orig_plate:
            # Try to extract from front/rear images
            for side in ["front", "rear"]:
                img_path = sessions[session_id].get(side)
                if img_path:
                    plate, _ = detect_and_read_plate(img_path)
                    if plate:
                        sessions[session_id]["number_plate"] = plate
                        orig_plate = plate
                        break
        if orig_plate and plate_text:
            if plate_text.replace(" ","").lower() == orig_plate.replace(" ","").lower():
                return {
                    "status": "success",
                    "message": "Number plate matched successfully. This is the damaged vehicle.",
                    "number_plate": plate_text
                }
            else:
                return {
                    "status": "error",
                    "message": "Number plate mismatch. This is not the same vehicle.",
                    "number_plate": plate_text,
                    "original_number_plate": orig_plate
                }
        else:
            return {
                "status": "error",
                "message": "Could not extract number plate from original images for comparison.",
                "number_plate": plate_text
            }
    else:
        return {
            "status": "error",
            "message": f"Number plate extraction failed: {err}",
            "error_code": "PLATE_EXTRACTION_FAILED"
        }

@app.post("/upload-towing/{session_id}")
async def upload_towing_vehicle(
    session_id: str,
    file: UploadFile = File(...),
):
    """
    Upload an image of the towing service vehicle, extract and compare number plate.
    """
    if not is_valid_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    # Save file
    session_path = get_session_path(session_id)
    temp_file_path = session_path / f"towing_{uuid.uuid4()}.jpg"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Detect vehicle
    vehicle_detected = False
    if yolo_model is not None:
        results = yolo_model(str(temp_file_path))
        labels = results.pandas().xyxy[0]['name'].tolist()
        if any(label in ['car', 'motorcycle', 'bus', 'truck'] for label in labels):
            vehicle_detected = True
    else:
        vehicle_detected = True
    if not vehicle_detected:
        temp_file_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "No vehicle is detected in the towing image.",
                "error_code": "NO_VEHICLE"
            }
        )
    # Detect and read number plate
    plate_text, err = detect_and_read_plate(temp_file_path)
    if plate_text:
        sessions[session_id]["towing_number_plate"] = plate_text
        damaged_plate = sessions[session_id].get("towed_number_plate")
        if damaged_plate and plate_text.replace(" ","").lower() == damaged_plate.replace(" ","").lower():
            return {
                "status": "error",
                "message": "Error: Towing vehicle number plate matches the damaged vehicle.",
                "number_plate": plate_text
            }
        else:
            return {
                "status": "success",
                "message": "Towing service vehicle number plate extracted successfully. It is not the damaged vehicle.",
                "number_plate": plate_text
            }
    else:
        return {
            "status": "error",
            "message": f"Number plate extraction failed: {err}",
            "error_code": "PLATE_EXTRACTION_FAILED"
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 