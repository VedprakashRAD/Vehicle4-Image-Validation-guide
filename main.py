from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import uuid
import shutil
from PIL import Image
import imagehash
import numpy as np
from pathlib import Path
import uvicorn

# Import our custom modules
from ml_model import VehicleSideClassifier
from image_utils import calculate_image_hash, is_duplicate, check_image_quality

# Create app instance
app = FastAPI(title="Vehicle Image Validation API")

# Add CORS middleware to allow requests from Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your Flutter app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory storage for session data (replace with database in production)
sessions = {}

# Initialize the vehicle side classifier
classifier = VehicleSideClassifier()

def get_session_path(session_id: str) -> Path:
    """Get the path for a session's uploads"""
    session_path = UPLOAD_DIR / session_id
    session_path.mkdir(exist_ok=True)
    return session_path

def create_session():
    """Create a new session for tracking uploads"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "front": None,
        "rear": None,
        "left": None, 
        "right": None
    }
    return session_id

@app.get("/")
def read_root():
    return {"message": "Vehicle Image Validation API"}

@app.post("/session")
def create_new_session():
    """Create a new upload session"""
    session_id = create_session()
    return {"session_id": session_id}

@app.post("/upload/{session_id}")
async def upload_image(
    session_id: str,
    file: UploadFile = File(...),
    claimed_side: str = Form(...),
):
    """Upload and validate a vehicle image"""
    # Check if session exists
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate claimed side
    if claimed_side not in ["front", "rear", "left", "right"]:
        raise HTTPException(status_code=400, detail="Invalid side. Must be front, rear, left, or right")
    
    # Save the uploaded file
    session_path = get_session_path(session_id)
    temp_file_path = session_path / f"{uuid.uuid4()}.jpg"
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image
    try:
        # Open image for processing
        img = Image.open(temp_file_path)
        
        # Check image quality
        quality_ok, quality_message = check_image_quality(img)
        if not quality_ok:
            os.remove(temp_file_path)
            return {
                "status": "error",
                "message": quality_message
            }
        
        # Calculate image hash for duplicate detection
        img_hash = calculate_image_hash(img)
        
        # Get existing hashes from this session
        existing_hashes = []
        for side, info in sessions[session_id].items():
            if info and "hash" in info:
                existing_hashes.append(info["hash"])
        
        # Check for duplicates
        is_dup, matching_hash = is_duplicate(img, existing_hashes)
        if is_dup:
            # Find which side the duplicate is from
            duplicate_side = None
            for side, info in sessions[session_id].items():
                if info and info.get("hash") == matching_hash:
                    duplicate_side = side
                    break
                    
            os.remove(temp_file_path)
            return {
                "status": "error",
                "message": f"This appears to be a duplicate of the {duplicate_side} image"
            }
        
        # In a real application, use the classifier to detect the side
        # For now, we'll use the claimed side but simulate classification
        detected_side = classifier.predict(img)
        
        # For demo purposes, we'll trust the user's claimed side 80% of the time
        # In a real app, you would use the actual classifier output
        if np.random.random() > 0.8:
            detected_side = claimed_side
        
        # If the detected side doesn't match the claimed side
        if detected_side != claimed_side:
            os.remove(temp_file_path)
            return {
                "status": "error",
                "message": f"This appears to be a {detected_side} view, not a {claimed_side} view"
            }
        
        # Check if this side was already uploaded
        if sessions[session_id][detected_side] is not None:
            os.remove(temp_file_path)
            return {
                "status": "error",
                "message": f"An image for the {detected_side} side was already uploaded"
            }
        
        # Store the image info
        final_path = session_path / f"{detected_side}.jpg"
        os.rename(temp_file_path, final_path)
        
        sessions[session_id][detected_side] = {
            "path": str(final_path),
            "hash": img_hash
        }
        
        # Check completion status
        missing_sides = [side for side, info in sessions[session_id].items() if info is None]
        
        return {
            "status": "success",
            "detected_side": detected_side,
            "missing_sides": missing_sides,
            "is_complete": len(missing_sides) == 0
        }
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

@app.get("/status/{session_id}")
def check_status(session_id: str):
    """Check the status of uploaded images for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = sessions[session_id]
    uploaded_sides = [side for side, info in session_data.items() if info is not None]
    missing_sides = [side for side, info in session_data.items() if info is None]
    
    return {
        "uploaded_sides": uploaded_sides,
        "missing_sides": missing_sides,
        "is_complete": len(missing_sides) == 0
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 