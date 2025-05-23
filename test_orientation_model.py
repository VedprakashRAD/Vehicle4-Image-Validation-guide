import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse
from PIL import Image, ImageDraw, ImageFont

def load_orientation_model(model_path):
    """Load the trained orientation model"""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
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

def predict_orientation(model, image_path, class_names):
    """Predict the orientation of a vehicle in an image"""
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img)[0]
    
    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions

def visualize_prediction(image_path, predicted_class, confidence, class_names, predictions, output_path=None):
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
    text = f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}"
    draw.text((10, 10), text, fill=(255, 0, 0), font=font)
    
    # Convert back to numpy array
    img_with_text = np.array(pil_img)
    
    # Create a figure with the image and a bar chart of predictions
    plt.figure(figsize=(12, 6))
    
    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(img_with_text)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
    plt.axis('off')
    
    # Plot the prediction probabilities
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, predictions)
    plt.yticks(y_pos, class_names)
    plt.xlabel('Probability')
    plt.title('Orientation Probabilities')
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def test_on_directory(model, test_dir, class_names, output_dir=None):
    """Test the model on all images in a directory"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {test_dir}")
        return
    
    print(f"Testing on {len(image_files)} images...")
    
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        print(f"\nProcessing {img_path}...")
        
        try:
            # Predict orientation
            predicted_class, confidence, predictions = predict_orientation(model, img_path, class_names)
            print(f"Predicted: {predicted_class}, Confidence: {confidence:.4f}")
            
            # Visualize prediction
            if output_dir:
                output_path = os.path.join(output_dir, f"pred_{os.path.splitext(img_file)[0]}.png")
                visualize_prediction(img_path, predicted_class, confidence, class_names, predictions, output_path)
            else:
                visualize_prediction(img_path, predicted_class, confidence, class_names, predictions)
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test vehicle orientation model on images")
    parser.add_argument("--model", default="orientation_model_resnet50/best_model.h5", help="Path to the trained model")
    parser.add_argument("--test_dir", default="test_images", help="Directory containing test images")
    parser.add_argument("--output_dir", default="test_results", help="Directory to save visualization results")
    args = parser.parse_args()
    
    # Define class names (should match the training order)
    class_names = [
        "front_view",
        "left_side_view",
        "rear_view",
        "right_side_view"
    ]
    
    # Load the model
    model = load_orientation_model(args.model)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Test on directory
    test_on_directory(model, args.test_dir, class_names, args.output_dir)

if __name__ == "__main__":
    main() 