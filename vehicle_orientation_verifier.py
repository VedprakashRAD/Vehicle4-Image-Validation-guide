import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from PIL import Image, ImageDraw, ImageFont

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

def load_orientation_model(model_path=MODEL_PATH):
    """Load the trained orientation model"""
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_orientation(model, image_path):
    """Predict the orientation of a vehicle in an image"""
    # Preprocess the image
    img = preprocess_image(image_path)
    if img is None:
        return None, 0, []
    
    # Make prediction
    predictions = model.predict(img)[0]
    
    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    # Get user-friendly orientation name
    orientation = CLASS_MAPPING.get(predicted_class, predicted_class)
    
    return orientation, confidence, predictions

def verify_orientation(model, image_path, expected_orientation):
    """Verify if the image has the expected vehicle orientation"""
    # Get the expected orientation in standard format
    expected_std = None
    for class_name, friendly_name in CLASS_MAPPING.items():
        if friendly_name.lower() == expected_orientation.lower():
            expected_std = friendly_name
            break
    
    if expected_std is None:
        return {
            "is_valid": False,
            "message": f"Invalid expected orientation: {expected_orientation}. Must be one of: Front, Rear, Left, Right"
        }
    
    # Predict orientation
    detected_orientation, confidence, _ = predict_orientation(model, image_path)
    
    if detected_orientation is None:
        return {
            "is_valid": False,
            "message": "Could not process the image"
        }
    
    # Check if confidence is too low - using a lower threshold since our model has lower confidence values
    if confidence < 0.35:
        return {
            "is_valid": False,
            "message": "No vehicle is detected or orientation is unclear",
            "confidence": float(confidence)
        }
    
    # Check if orientation matches expected
    if detected_orientation.lower() == expected_orientation.lower():
        return {
            "is_valid": True,
            "message": f"{detected_orientation} side uploaded successfully",
            "confidence": float(confidence)
        }
    else:
        return {
            "is_valid": False,
            "message": f"Mismatch: {detected_orientation} side is detected for the intended {expected_orientation} side",
            "confidence": float(confidence)
        }

def visualize_verification(image_path, result, output_path=None):
    """Visualize the verification result on the image"""
    # Read the original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a PIL image for adding text
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Set text color based on validation result
    text_color = (0, 255, 0) if result.get("is_valid", False) else (255, 0, 0)
    
    # Add result text - using ASCII characters instead of Unicode
    text = f"Result: {'PASS' if result.get('is_valid', False) else 'FAIL'}\n{result.get('message', '')}"
    if "confidence" in result:
        text += f"\nConfidence: {result.get('confidence', 0):.2f}"
    
    draw.text((10, 10), text, fill=text_color, font=font)
    
    # Convert back to numpy array
    img_with_text = np.array(pil_img)
    
    # Save or display the image
    if output_path:
        # Convert RGB back to BGR for OpenCV
        img_with_text = cv2.cvtColor(img_with_text, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_with_text)
        print(f"Visualization saved to {output_path}")
    else:
        # Display using matplotlib
        import matplotlib.pyplot as plt
        plt.imshow(img_with_text)
        plt.axis('off')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Vehicle Orientation Verification Tool")
    parser.add_argument("--image", required=True, help="Path to the vehicle image")
    parser.add_argument("--expected", required=True, choices=["front", "rear", "left", "right"], 
                        help="Expected orientation (front, rear, left, right)")
    parser.add_argument("--output", help="Path to save visualization (optional)")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the trained model")
    
    args = parser.parse_args()
    
    # Load the model
    model = load_orientation_model(args.model)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Verify orientation
    result = verify_orientation(model, args.image, args.expected)
    print("\nVerification Result:")
    print(f"Valid: {result['is_valid']}")
    print(f"Message: {result['message']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    
    # Visualize the result
    if args.output:
        visualize_verification(args.image, result, args.output)
    else:
        visualize_verification(args.image, result)

if __name__ == "__main__":
    main() 