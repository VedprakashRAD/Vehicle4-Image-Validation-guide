from PIL import Image
import imagehash
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
import tensorflow as tf
import math

def calculate_image_hash(img: Image.Image) -> str:
    """
    Calculate a perceptual hash for an image to detect duplicates
    
    Args:
        img: PIL Image object
        
    Returns:
        str: The hash value as a string
    """
    return str(imagehash.phash(img))

def is_duplicate(img: Image.Image, existing_hashes: List[str], threshold: int = 5) -> Tuple[bool, Optional[str]]:
    """
    Check if an image is a duplicate of any existing images
    
    Args:
        img: PIL Image object
        existing_hashes: List of hash values from existing images
        threshold: Maximum hash difference to consider as duplicate (lower = stricter)
        
    Returns:
        Tuple[bool, Optional[str]]: (is_duplicate, matching_hash)
    """
    img_hash = imagehash.phash(img)
    
    for hash_str in existing_hashes:
        existing_hash = imagehash.hex_to_hash(hash_str)
        if img_hash - existing_hash < threshold:
            return True, hash_str
            
    return False, None

def detect_vehicle(img: Image.Image, model=None, confidence_threshold: float = 0.5) -> Tuple[bool, float, Dict]:
    """
    Detect if a vehicle is present in the image using object detection
    
    Args:
        img: PIL Image object
        model: Object detection model (uses TF Hub model if None)
        confidence_threshold: Minimum confidence score to consider a detection valid
        
    Returns:
        Tuple[bool, float, Dict]: (vehicle_present, confidence, bounding_box)
    """
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # If no model is provided, use a simple heuristic method
    # In a real application, you would use a proper object detection model like YOLOv5 or SSD
    if model is None:
        # Use simple edge detection and contour analysis as a placeholder
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges
            edges = cv2.Canny(blur, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            large_contours = [c for c in contours if cv2.contourArea(c) > img.width * img.height * 0.1]
            
            if large_contours:
                # Find the largest contour
                largest_contour = max(large_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Calculate aspect ratio (vehicles typically have specific aspect ratios)
                aspect_ratio = w / h
                
                # Check if the aspect ratio is reasonable for a vehicle (rough heuristic)
                if 0.5 <= aspect_ratio <= 3.0:
                    confidence = cv2.contourArea(largest_contour) / (img.width * img.height)
                    bounding_box = {"x": x, "y": y, "width": w, "height": h}
                    return True, confidence, bounding_box
            
            return False, 0.0, {}
            
        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            return False, 0.0, {}
    
    # If a model is provided, use it for object detection
    else:
        try:
            # Implement model-specific preprocessing and inference
            # For now, return a placeholder
            return True, 0.7, {"x": 100, "y": 100, "width": 300, "height": 200}
        except Exception as e:
            print(f"Error in model-based vehicle detection: {e}")
            return False, 0.0, {}

def check_image_quality(img: Image.Image, min_width: int = 800, min_height: int = 600) -> Tuple[bool, str]:
    """
    Check if an image meets minimum quality requirements
    
    Args:
        img: PIL Image object
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels
        
    Returns:
        Tuple[bool, str]: (is_acceptable, reason)
    """
    width, height = img.size
    
    # Check dimensions
    if width < min_width or height < min_height:
        return False, f"Image too small. Minimum size is {min_width}x{min_height} pixels."
    
    # Check if image is not too blurry
    blurry, blur_msg = check_image_blur(img)
    if blurry:
        return False, blur_msg
    
    # Check lighting conditions
    good_lighting, lighting_msg = check_lighting_conditions(img)
    if not good_lighting:
        return False, lighting_msg
    
    # Check if a vehicle is detected in the image
    vehicle_detected, _, _ = detect_vehicle(img)
    if not vehicle_detected:
        return False, "No vehicle detected in the image."
        
    return True, "Image quality acceptable"

def check_image_blur(img: Image.Image, threshold: float = 100.0) -> Tuple[bool, str]:
    """
    Check if an image is too blurry using Laplacian variance
    
    Args:
        img: PIL Image object
        threshold: Minimum Laplacian variance for acceptable sharpness
        
    Returns:
        Tuple[bool, str]: (is_blurry, message)
    """
    try:
        # Convert to grayscale
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # Calculate a normalized blur score based on image size
        size_factor = math.sqrt(img.width * img.height) / 1000  # Normalize by image size
        adjusted_threshold = threshold * size_factor
        
        if laplacian_var < adjusted_threshold:
            return True, f"Image appears to be blurry (score: {laplacian_var:.2f}, threshold: {adjusted_threshold:.2f})."
            
        return False, "Image sharpness is acceptable."
    except Exception as e:
        print(f"Error in blur detection: {e}")
        return False, "Unable to check blur, assuming image is acceptable."

def check_lighting_conditions(img: Image.Image) -> Tuple[bool, str]:
    """
    Check if an image has good lighting conditions (not too dark or too bright)
    
    Args:
        img: PIL Image object
        
    Returns:
        Tuple[bool, str]: (has_good_lighting, message)
    """
    try:
        # Convert to grayscale and get pixel values
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # Calculate brightness statistics
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)
        
        # Check if image is too dark
        if mean_brightness < 40:
            return False, f"Image is too dark (brightness: {mean_brightness:.2f})."
            
        # Check if image is too bright
        if mean_brightness > 220:
            return False, f"Image is too bright (brightness: {mean_brightness:.2f})."
            
        # Check if image has enough contrast
        if std_brightness < 20:
            return False, f"Image has low contrast (std: {std_brightness:.2f})."
            
        return True, "Lighting conditions are acceptable."
    except Exception as e:
        print(f"Error in lighting check: {e}")
        return True, "Unable to check lighting, assuming conditions are acceptable."

def check_vehicle_perspective(img: Image.Image, claimed_side: str) -> Tuple[bool, float, str]:
    """
    Check if the image shows the vehicle from the correct perspective
    
    Args:
        img: PIL Image object
        claimed_side: The side claimed by the user ("front", "rear", "left", "right")
        
    Returns:
        Tuple[bool, float, str]: (is_correct_perspective, confidence, message)
    """
    # This is a placeholder for perspective checking
    # In a real app, you would use a trained vehicle side classifier
    
    # For now, return a placeholder result based on basic image analysis
    try:
        # Convert to grayscale
        img_array = np.array(img)
        
        # Detect edges for shape analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density in different regions of the image
        h, w = edges.shape
        left_density = np.count_nonzero(edges[:, :w//3]) / (h * w//3)
        center_density = np.count_nonzero(edges[:, w//3:2*w//3]) / (h * w//3)
        right_density = np.count_nonzero(edges[:, 2*w//3:]) / (h * w//3)
        
        top_density = np.count_nonzero(edges[:h//3, :]) / (h//3 * w)
        middle_density = np.count_nonzero(edges[h//3:2*h//3, :]) / (h//3 * w)
        bottom_density = np.count_nonzero(edges[2*h//3:, :]) / (h//3 * w)
        
        # Simple heuristics for perspective (these are just placeholders)
        perspective_confidence = 0.7  # Default confidence
        
        if claimed_side == "front":
            if center_density > left_density and center_density > right_density:
                return True, perspective_confidence, "Image appears to show the front of the vehicle."
            else:
                return False, 0.4, "Image may not show the front of the vehicle clearly."
                
        elif claimed_side == "rear":
            if center_density > left_density and center_density > right_density:
                return True, perspective_confidence, "Image appears to show the rear of the vehicle."
            else:
                return False, 0.4, "Image may not show the rear of the vehicle clearly."
                
        elif claimed_side == "left":
            if left_density < right_density:
                return True, perspective_confidence, "Image appears to show the left side of the vehicle."
            else:
                return False, 0.4, "Image may not show the left side of the vehicle clearly."
                
        elif claimed_side == "right":
            if right_density < left_density:
                return True, perspective_confidence, "Image appears to show the right side of the vehicle."
            else:
                return False, 0.4, "Image may not show the right side of the vehicle clearly."
        
        # Default fallback
        return True, 0.5, f"Unable to definitively confirm {claimed_side} perspective."
        
    except Exception as e:
        print(f"Error in perspective check: {e}")
        return True, 0.5, "Unable to check perspective, assuming it's acceptable."

def check_full_vehicle_visible(img: Image.Image) -> Tuple[bool, str]:
    """
    Check if the full vehicle is visible in the frame
    
    Args:
        img: PIL Image object
        
    Returns:
        Tuple[bool, str]: (is_fully_visible, message)
    """
    # Detect vehicle
    vehicle_detected, confidence, bbox = detect_vehicle(img)
    
    if not vehicle_detected:
        return False, "No vehicle detected in the image."
    
    # If bbox is empty, return False
    if not bbox:
        return False, "Could not determine vehicle boundaries."
    
    # Check if the bounding box extends to the edges of the image
    margin = 20  # Pixel margin from the edge
    
    width, height = img.size
    x, y = bbox.get("x", 0), bbox.get("y", 0)
    w, h = bbox.get("width", 0), bbox.get("height", 0)
    
    # Check if the bounding box is too close to the edges
    if x < margin or y < margin or x + w > width - margin or y + h > height - margin:
        return False, "Vehicle appears to be cut off at the edge of the frame."
    
    # Check if the bounding box is of reasonable size relative to the image
    bbox_area = w * h
    img_area = width * height
    bbox_ratio = bbox_area / img_area
    
    if bbox_ratio < 0.2:
        return False, "Vehicle appears too small in the frame."
    
    if bbox_ratio > 0.9:
        return False, "Vehicle appears too close to the camera."
    
    return True, "Full vehicle appears to be visible in the frame."

def preprocess_for_model(img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess an image for input to a machine learning model
    
    Args:
        img: PIL Image object
        target_size: Size to resize the image to
        
    Returns:
        np.ndarray: Preprocessed image as numpy array
    """
    # Resize image
    img_resized = img.resize(target_size)
    
    # Convert to RGB if not already
    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img_resized) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def comprehensive_image_check(
    img: Image.Image, 
    claimed_side: str, 
    min_width: int = 800, 
    min_height: int = 600
) -> Tuple[bool, Dict]:
    """
    Run a comprehensive check on the image quality and vehicle visibility
    
    Args:
        img: PIL Image object
        claimed_side: The side claimed by the user ("front", "rear", "left", "right")
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels
        
    Returns:
        Tuple[bool, Dict]: (is_acceptable, detailed_results)
    """
    results = {}
    
    # Check dimensions
    width, height = img.size
    results["dimensions"] = {
        "is_acceptable": width >= min_width and height >= min_height,
        "width": width,
        "height": height,
        "min_width": min_width,
        "min_height": min_height,
        "message": f"Image size: {width}x{height}" if width >= min_width and height >= min_height else f"Image too small. Minimum size is {min_width}x{min_height}."
    }
    
    # Check blur
    is_blurry, blur_msg = check_image_blur(img)
    results["blur"] = {
        "is_acceptable": not is_blurry,
        "message": blur_msg
    }
    
    # Check lighting
    good_lighting, lighting_msg = check_lighting_conditions(img)
    results["lighting"] = {
        "is_acceptable": good_lighting,
        "message": lighting_msg
    }
    
    # Check vehicle detection
    vehicle_detected, confidence, bbox = detect_vehicle(img)
    results["vehicle_detection"] = {
        "is_acceptable": vehicle_detected,
        "confidence": confidence,
        "bbox": bbox,
        "message": "Vehicle detected" if vehicle_detected else "No vehicle detected in the image."
    }
    
    # Check perspective
    correct_perspective, perspective_confidence, perspective_msg = check_vehicle_perspective(img, claimed_side)
    results["perspective"] = {
        "is_acceptable": correct_perspective,
        "confidence": perspective_confidence,
        "message": perspective_msg
    }
    
    # Check if full vehicle is visible
    fully_visible, visibility_msg = check_full_vehicle_visible(img)
    results["visibility"] = {
        "is_acceptable": fully_visible,
        "message": visibility_msg
    }
    
    # Overall result
    is_acceptable = (
        results["dimensions"]["is_acceptable"] and
        results["blur"]["is_acceptable"] and
        results["lighting"]["is_acceptable"] and
        results["vehicle_detection"]["is_acceptable"] and
        results["perspective"]["is_acceptable"] and
        results["visibility"]["is_acceptable"]
    )
    
    return is_acceptable, results 