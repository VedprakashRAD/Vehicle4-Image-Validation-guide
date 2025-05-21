from PIL import Image
import imagehash
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2

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
    
    # Check if image is not too blurry (simplified)
    # In a real app, you would use more sophisticated blur detection
    try:
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        laplacian_var = np.var(cv2.Laplacian(img_array, cv2.CV_64F))
        
        if laplacian_var < 100:  # Arbitrary threshold
            return False, "Image appears to be blurry."
    except:
        # If we can't check blur (e.g., cv2 not available), skip this check
        pass
    
    return True, "Image quality acceptable"

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