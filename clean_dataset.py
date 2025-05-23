import os
import cv2
from PIL import Image
import shutil
from tqdm import tqdm

def is_valid_image(img_path):
    """Check if an image is valid using both OpenCV and PIL"""
    try:
        # Try with OpenCV
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            return False
        
        # Also try with PIL
        with Image.open(img_path) as img:
            img.verify()
        
        return True
    except Exception as e:
        print(f"Invalid image {img_path}: {str(e)}")
        return False

def clean_dataset(dataset_dir):
    """Remove corrupted images from the dataset"""
    print(f"Cleaning dataset in {dataset_dir}...")
    
    total_images = 0
    removed_images = 0
    
    # Process each orientation directory
    for orientation in os.listdir(dataset_dir):
        orientation_dir = os.path.join(dataset_dir, orientation)
        if not os.path.isdir(orientation_dir):
            continue
        
        print(f"\nProcessing {orientation} directory...")
        
        # Get all image files
        image_files = [
            f for f in os.listdir(orientation_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        total_images += len(image_files)
        print(f"Found {len(image_files)} images")
        
        # Check each image
        for img_file in tqdm(image_files, desc=f"Checking {orientation} images"):
            img_path = os.path.join(orientation_dir, img_file)
            if not is_valid_image(img_path):
                # Remove corrupted image
                os.remove(img_path)
                removed_images += 1
    
    print(f"\nDataset cleaning complete!")
    print(f"Total images: {total_images}")
    print(f"Removed images: {removed_images}")
    print(f"Remaining images: {total_images - removed_images}")

def main():
    # Clean the processed dataset
    clean_dataset("processed_large_dataset")

if __name__ == "__main__":
    main() 