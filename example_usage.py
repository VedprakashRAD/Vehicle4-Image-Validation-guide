import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import glob

# Define constants
MODEL_PATH = "orientation_model_resnet50/best_model.h5"
CLASS_NAMES = [
    "front_view",
    "left_side_view",
    "rear_view",
    "right_side_view"
]

# User-friendly names for orientations
ORIENTATION_NAMES = {
    "front_view": "Front",
    "rear_view": "Rear",
    "left_side_view": "Left",
    "right_side_view": "Right"
}

def load_orientation_model():
    """Load the trained orientation model"""
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for prediction"""
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

def predict_orientation(model, image_path):
    """Predict the orientation of a vehicle in an image"""
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img)[0]
    
    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    return predicted_class, confidence, predictions

def verify_orientation(model, image_path, expected_orientation):
    """Verify if the image has the expected vehicle orientation"""
    # Predict orientation
    predicted_class, confidence, _ = predict_orientation(model, image_path)
    
    # Get user-friendly names
    predicted_orientation = ORIENTATION_NAMES.get(predicted_class, predicted_class)
    expected_orientation_friendly = expected_orientation.capitalize()
    
    # Check if confidence is too low (no vehicle detected)
    if confidence < 0.4:
        return {
            "is_valid": False,
            "message": "No vehicle is detected or orientation is unclear",
            "confidence": float(confidence)
        }
    
    # Check if orientation matches expected
    if predicted_orientation.lower() == expected_orientation.lower():
        return {
            "is_valid": True,
            "message": f"{predicted_orientation} side uploaded successfully",
            "confidence": float(confidence)
        }
    else:
        return {
            "is_valid": False,
            "message": f"Mismatch: {predicted_orientation} side is detected for the intended {expected_orientation_friendly} side",
            "confidence": float(confidence)
        }

def visualize_prediction(image_path, result):
    """Visualize the prediction on the image"""
    # Read the original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a PIL image for adding text
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Add prediction text
    text = f"Result: {'✓' if result['is_valid'] else '✗'}\n{result['message']}\nConfidence: {result['confidence']:.2f}"
    draw.text((10, 10), text, fill=(255, 0, 0), font=font)
    
    # Convert back to numpy array
    img_with_text = np.array(pil_img)
    
    # Display the image
    plt.imshow(img_with_text)
    plt.axis('off')
    plt.show()

def main():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Train the model first.")
        return
    
    # Load the model
    model = load_orientation_model()
    
    # Get test images
    test_images = glob.glob("test_images/*.jpg")
    
    if not test_images:
        print("No test images found in the test_images directory.")
        return
    
    print(f"Found {len(test_images)} test images.")
    
    # Example 1: Verify orientation
    print("\nExample 1: Verifying orientation")
    test_image = test_images[0]
    expected_orientation = "front"  # This is just an example
    
    print(f"Verifying if {test_image} is a {expected_orientation} view...")
    result = verify_orientation(model, test_image, expected_orientation)
    print(f"Result: {result}")
    
    # Visualize the result
    visualize_prediction(test_image, result)
    
    # Example 2: Batch processing
    print("\nExample 2: Batch processing")
    print("Processing all test images...")
    
    for i, image_path in enumerate(test_images[:5]):  # Process first 5 images
        predicted_class, confidence, _ = predict_orientation(model, image_path)
        orientation = ORIENTATION_NAMES.get(predicted_class, predicted_class)
        print(f"Image {i+1}: {os.path.basename(image_path)}")
        print(f"  Predicted: {orientation} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main() 