import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
import shutil
from tqdm import tqdm

def create_processed_directory(base_dir="vehicle_orientation_dataset", processed_dir="processed_dataset"):
    """Create directory for processed images"""
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # Create subdirectories for each orientation
    orientations = os.listdir(base_dir)
    for orientation in orientations:
        orientation_path = os.path.join(base_dir, orientation)
        if os.path.isdir(orientation_path):
            processed_orientation_path = os.path.join(processed_dir, orientation)
            if not os.path.exists(processed_orientation_path):
                os.makedirs(processed_orientation_path)
    
    return processed_dir

def resize_image(image, target_size=(224, 224)):
    """Resize image to target size"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(image):
    """Normalize pixel values to [0, 1]"""
    return image.astype(np.float32) / 255.0

def center_crop(image, crop_size=(224, 224)):
    """Center crop the image to the specified size"""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    
    return image[start_h:end_h, start_w:end_w]

def random_crop(image, crop_size=(224, 224)):
    """Random crop the image to the specified size"""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    
    if h <= crop_h or w <= crop_w:
        return resize_image(image, crop_size)
    
    start_h = random.randint(0, h - crop_h)
    start_w = random.randint(0, w - crop_w)
    
    end_h = start_h + crop_h
    end_w = start_w + crop_w
    
    return image[start_h:end_h, start_w:end_w]

def adjust_brightness(image, factor):
    """Adjust brightness of the image"""
    pil_image = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_image)
    adjusted = enhancer.enhance(factor)
    return np.array(adjusted)

def adjust_contrast(image, factor):
    """Adjust contrast of the image"""
    pil_image = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_image)
    adjusted = enhancer.enhance(factor)
    return np.array(adjusted)

def add_noise(image, noise_level=0.05):
    """Add random noise to the image"""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1)
    return (noisy_image * 255).astype(np.uint8)

def augment_image(image):
    """Apply random augmentations to the image"""
    augmentations = []
    
    # Brightness adjustment
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        augmentations.append(adjust_brightness(image, factor))
    
    # Contrast adjustment
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.3)
        augmentations.append(adjust_contrast(image, factor))
    
    # Random crop and resize
    if random.random() < 0.5:
        augmentations.append(resize_image(random_crop(image, (int(image.shape[0] * 0.8), int(image.shape[1] * 0.8)))))
    
    # Add noise
    if random.random() < 0.3:
        augmentations.append(add_noise(image))
    
    # If no augmentations were applied, return the original image
    if not augmentations:
        return [image]
    
    return augmentations

def process_dataset(base_dir="vehicle_orientation_dataset", processed_dir="processed_dataset", augment=True):
    """Process and augment the dataset"""
    print(f"Processing dataset from {base_dir} to {processed_dir}...")
    
    # Create processed directory
    processed_dir = create_processed_directory(base_dir, processed_dir)
    
    # Process each orientation
    orientations = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    total_original = 0
    total_processed = 0
    
    for orientation in orientations:
        print(f"\nProcessing {orientation}...")
        orientation_path = os.path.join(base_dir, orientation)
        processed_orientation_path = os.path.join(processed_dir, orientation)
        
        # Get all images in this orientation
        images = [f for f in os.listdir(orientation_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        total_original += len(images)
        
        # Process each image
        for img_file in tqdm(images, desc=f"Processing {orientation}"):
            img_path = os.path.join(orientation_path, img_file)
            
            try:
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize image
                image = resize_image(image)
                
                # Save the processed original image
                processed_img_path = os.path.join(processed_orientation_path, img_file)
                cv2.imwrite(processed_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                total_processed += 1
                
                # Apply augmentations if enabled
                if augment:
                    augmented_images = augment_image(image)
                    
                    # Save augmented images
                    for i, aug_img in enumerate(augmented_images):
                        aug_filename = f"{os.path.splitext(img_file)[0]}_aug{i}{os.path.splitext(img_file)[1]}"
                        aug_path = os.path.join(processed_orientation_path, aug_filename)
                        cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                        total_processed += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Original images: {total_original}")
    print(f"Processed images (including augmentations): {total_processed}")
    print(f"Processed dataset saved to: {processed_dir}")

def main():
    # Process the dataset
    process_dataset(base_dir="vehicle_orientation_dataset", processed_dir="processed_dataset", augment=True)

if __name__ == "__main__":
    main() 