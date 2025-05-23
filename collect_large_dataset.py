from simple_image_download import simple_image_download as simp
import os
import time
import random
import shutil
import requests
from tqdm import tqdm
import concurrent.futures
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dataset_collection.log'
)

def create_directories():
    """Create directories for each orientation category"""
    orientations = [
        "front_view", 
        "rear_view", 
        "left_side_view", 
        "right_side_view"
    ]
    
    base_dir = "large_vehicle_dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for orientation in orientations:
        orientation_dir = os.path.join(base_dir, orientation)
        if not os.path.exists(orientation_dir):
            os.makedirs(orientation_dir)
    
    return base_dir, orientations

def generate_search_terms():
    """Generate diverse search terms for each orientation"""
    vehicle_types = [
        "car", "automobile", "sedan", "SUV", "truck", "pickup truck",
        "motorcycle", "bike", "motorbike", 
        "scooter", "moped", "vespa"
    ]
    
    colors = ["red", "blue", "black", "white", "silver", "green", "yellow"]
    
    search_terms = {
        "front_view": [],
        "rear_view": [],
        "left_side_view": [],
        "right_side_view": []
    }
    
    # Generate front view terms
    for vehicle in vehicle_types:
        search_terms["front_view"].extend([
            f"{vehicle} front view",
            f"{vehicle} front angle",
            f"{vehicle} front photo",
            f"{vehicle} front picture",
        ])
        
        # Add some with colors
        for color in colors[:3]:  # Use first few colors
            search_terms["front_view"].append(f"{color} {vehicle} front view")
    
    # Generate rear view terms
    for vehicle in vehicle_types:
        search_terms["rear_view"].extend([
            f"{vehicle} rear view",
            f"{vehicle} back view",
            f"{vehicle} rear angle",
            f"{vehicle} back photo",
            f"{vehicle} rear picture",
        ])
        
        # Add some with colors
        for color in colors[:3]:  # Use first few colors
            search_terms["rear_view"].append(f"{color} {vehicle} rear view")
    
    # Generate left side view terms
    for vehicle in vehicle_types:
        search_terms["left_side_view"].extend([
            f"{vehicle} left side view",
            f"{vehicle} left side",
            f"{vehicle} left profile",
            f"{vehicle} left angle",
            f"{vehicle} driver side",
        ])
        
        # Add some with colors
        for color in colors[:3]:  # Use first few colors
            search_terms["left_side_view"].append(f"{color} {vehicle} left side")
    
    # Generate right side view terms
    for vehicle in vehicle_types:
        search_terms["right_side_view"].extend([
            f"{vehicle} right side view",
            f"{vehicle} right side",
            f"{vehicle} right profile",
            f"{vehicle} right angle",
            f"{vehicle} passenger side",
        ])
        
        # Add some with colors
        for color in colors[:3]:  # Use first few colors
            search_terms["right_side_view"].append(f"{color} {vehicle} right side")
    
    return search_terms

def download_images_batch(orientation, search_term, images_per_term, response):
    """Download a batch of images for a specific search term"""
    try:
        print(f"  Downloading: '{search_term}'")
        response.download(search_term, images_per_term)
        return True
    except Exception as e:
        logging.error(f"Error downloading {search_term}: {str(e)}")
        return False

def download_images(orientations, target_count=1000):
    """Download images for each orientation"""
    print(f"Starting to download approximately {target_count} images for each orientation...")
    
    response = simp.simple_image_download()
    
    # Generate search terms
    search_terms = generate_search_terms()
    
    # Calculate images per term to reach target count
    for orientation in orientations:
        terms_count = len(search_terms[orientation])
        images_per_term = max(5, target_count // terms_count)
        
        print(f"\nDownloading images for {orientation}...")
        print(f"Using {terms_count} search terms with ~{images_per_term} images per term")
        
        # Use multithreading for faster downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for search_term in search_terms[orientation]:
                # Add some randomness to avoid exact same count
                term_count = images_per_term + random.randint(-2, 2)
                futures.append(
                    executor.submit(
                        download_images_batch, 
                        orientation, 
                        search_term, 
                        term_count, 
                        response
                    )
                )
                
                # Add a short delay between submissions to avoid rate limiting
                time.sleep(0.5)
            
            # Wait for all futures to complete with progress bar
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Downloading {orientation}"
            ):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Download task failed: {str(e)}")
    
    print("\nDownload completed!")
    return search_terms

def organize_images(base_dir, orientations, search_terms):
    """Move downloaded images to their respective orientation directories"""
    print("\nOrganizing downloaded images...")
    
    download_dir = "simple_images"
    if not os.path.exists(download_dir):
        print(f"Download directory '{download_dir}' not found.")
        return
    
    total_copied = 0
    
    for orientation in orientations:
        print(f"Processing {orientation} images...")
        orientation_search_terms = search_terms[orientation]
        
        # Move images from each search term folder to the orientation folder
        for search_term in tqdm(orientation_search_terms, desc=f"Organizing {orientation}"):
            # Directory name with spaces (as used by simple_image_download)
            search_term_dir = os.path.join(download_dir, search_term)
            if os.path.exists(search_term_dir):
                image_files = [f for f in os.listdir(search_term_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_file in image_files:
                    # Create a unique filename to avoid overwriting
                    src_path = os.path.join(search_term_dir, img_file)
                    dst_path = os.path.join(
                        base_dir, 
                        orientation, 
                        f"{search_term.replace(' ', '_')}_{total_copied}_{img_file}"
                    )
                    
                    # Copy the file
                    try:
                        shutil.copy2(src_path, dst_path)
                        total_copied += 1
                    except Exception as e:
                        logging.error(f"Error copying {src_path}: {str(e)}")
    
    print(f"Images organized successfully! Copied {total_copied} images.")
    return total_copied

def count_images_by_orientation(base_dir):
    """Count the number of images in each orientation directory"""
    counts = {}
    
    for orientation in os.listdir(base_dir):
        orientation_path = os.path.join(base_dir, orientation)
        if os.path.isdir(orientation_path):
            image_count = len([f for f in os.listdir(orientation_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            counts[orientation] = image_count
    
    return counts

def main():
    # Create directory structure
    base_dir, orientations = create_directories()
    
    # Download images - using a smaller target count for initial testing
    # Change this to 1000 when ready for the full dataset
    search_terms = download_images(orientations, target_count=50)
    
    # Organize images into orientation folders
    total_copied = organize_images(base_dir, orientations, search_terms)
    
    # Count images by orientation
    image_counts = count_images_by_orientation(base_dir)
    
    print("\nDataset collection complete!")
    print(f"Your dataset is available in the '{base_dir}' directory")
    print(f"Total images in dataset: {total_copied}")
    print("\nImages per orientation:")
    for orientation, count in image_counts.items():
        print(f"  {orientation}: {count} images")

if __name__ == "__main__":
    main() 