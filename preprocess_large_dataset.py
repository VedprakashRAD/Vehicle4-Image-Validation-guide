import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='preprocessing.log'
)

def create_output_directories(output_dir):
    """Create output directories for processed images"""
    orientations = [
        "front_view", 
        "rear_view", 
        "left_side_view", 
        "right_side_view"
    ]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for orientation in orientations:
        orientation_dir = os.path.join(output_dir, orientation)
        if not os.path.exists(orientation_dir):
            os.makedirs(orientation_dir)

def is_valid_image(img_path):
    """Check if an image is valid and can be opened"""
    try:
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            return False
        
        # Check image dimensions (exclude very small images)
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            return False
            
        return True
    except Exception as e:
        logging.error(f"Error validating image {img_path}: {str(e)}")
        return False

def process_image(args):
    """Process a single image with augmentations"""
    img_path, output_dir, orientation, idx, apply_augmentation = args
    
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Resize image to a standard size
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Generate base filename
        base_filename = f"{orientation}_{idx}"
        
        # Save the resized original image
        output_path = os.path.join(output_dir, orientation, f"{base_filename}_original.jpg")
        cv2.imwrite(output_path, img)
        
        # If augmentation is enabled, create augmented versions
        if apply_augmentation:
            # Apply random brightness and contrast adjustments
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            
            # Apply brightness adjustment
            bright_img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=brightness_factor * 10)
            output_path = os.path.join(output_dir, orientation, f"{base_filename}_bright.jpg")
            cv2.imwrite(output_path, bright_img)
            
            # Apply slight rotation (don't rotate too much as it can change the orientation)
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            output_path = os.path.join(output_dir, orientation, f"{base_filename}_rotated.jpg")
            cv2.imwrite(output_path, rotated_img)
            
            # Apply slight zoom (crop and resize)
            zoom_factor = random.uniform(0.8, 0.9)
            h, w = img.shape[:2]
            crop_h, crop_w = int(h * zoom_factor), int(w * zoom_factor)
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            cropped_img = img[start_h:start_h+crop_h, start_w:start_w+crop_w]
            zoomed_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
            output_path = os.path.join(output_dir, orientation, f"{base_filename}_zoomed.jpg")
            cv2.imwrite(output_path, zoomed_img)
            
            # Return the number of images created (original + augmented)
            return 4
        
        # If no augmentation, just return 1 for the original image
        return 1
    
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {str(e)}")
        return 0

def preprocess_dataset(input_dir, output_dir, limit_per_class=None, augment=True):
    """Preprocess the dataset with optional augmentation"""
    print(f"Preprocessing dataset from {input_dir} to {output_dir}...")
    
    # Create output directories
    create_output_directories(output_dir)
    
    # Get all orientation directories
    orientations = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    total_processed = 0
    total_augmented = 0
    
    # Process each orientation
    for orientation in orientations:
        print(f"\nProcessing {orientation} images...")
        orientation_dir = os.path.join(input_dir, orientation)
        
        # Get all image files
        image_files = [
            f for f in os.listdir(orientation_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # Limit the number of images if specified
        if limit_per_class and len(image_files) > limit_per_class:
            print(f"Limiting to {limit_per_class} images (from {len(image_files)} available)")
            random.shuffle(image_files)
            image_files = image_files[:limit_per_class]
        
        print(f"Found {len(image_files)} images for {orientation}")
        
        # Filter out invalid images
        valid_images = []
        for img_file in tqdm(image_files, desc="Validating images"):
            img_path = os.path.join(orientation_dir, img_file)
            if is_valid_image(img_path):
                valid_images.append(img_path)
            else:
                logging.warning(f"Skipping invalid image: {img_path}")
        
        print(f"Valid images: {len(valid_images)} out of {len(image_files)}")
        
        # Prepare arguments for parallel processing
        process_args = []
        for idx, img_path in enumerate(valid_images):
            process_args.append((img_path, output_dir, orientation, idx, augment))
        
        # Process images in parallel
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"Processing images using {num_cores} CPU cores...")
        
        results = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            for result in tqdm(
                executor.map(process_image, process_args),
                total=len(process_args),
                desc=f"Processing {orientation}"
            ):
                if result:
                    results.append(result)
        
        # Count processed and augmented images
        processed_count = len(valid_images)
        augmented_count = sum(results) if results else 0
        
        total_processed += processed_count
        total_augmented += augmented_count
        
        print(f"Processed {processed_count} images for {orientation}")
        print(f"Created {augmented_count} images (including originals and augmentations)")
    
    print("\nPreprocessing complete!")
    print(f"Total original images processed: {total_processed}")
    print(f"Total images after augmentation: {total_augmented}")
    
    return total_processed, total_augmented

def main():
    parser = argparse.ArgumentParser(description="Preprocess vehicle orientation dataset")
    parser.add_argument("--input_dir", default="large_vehicle_dataset", help="Input directory containing raw images")
    parser.add_argument("--output_dir", default="processed_large_dataset", help="Output directory for processed images")
    parser.add_argument("--limit", type=int, default=None, help="Limit images per class (None for all)")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    args = parser.parse_args()
    
    # Process the dataset
    preprocess_dataset(
        args.input_dir, 
        args.output_dir, 
        limit_per_class=args.limit,
        augment=not args.no_augment
    )

if __name__ == "__main__":
    main()

 