from simple_image_download import simple_image_download as simp
import os
import time
import random
import shutil

def create_directories():
    """Create directories for each orientation category"""
    orientations = [
        "front_view", 
        "rear_view", 
        "left_side_view", 
        "right_side_view"
    ]
    
    vehicle_types = ["car", "bike", "scooter"]
    
    base_dir = "vehicle_orientation_dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for orientation in orientations:
        orientation_dir = os.path.join(base_dir, orientation)
        if not os.path.exists(orientation_dir):
            os.makedirs(orientation_dir)
    
    return base_dir, orientations

def download_images(orientations, images_per_orientation=50):
    """Download images for each orientation"""
    print(f"Starting to download {images_per_orientation} images for each orientation...")
    
    response = simp.simple_image_download()
    
    # Add more specific search terms for better results
    search_terms = {
        "front_view": [
            "car front view", "automobile front view", "vehicle front view",
            "motorcycle front view", "bike front view", "scooter front view"
        ],
        "rear_view": [
            "car rear view", "automobile back view", "vehicle rear view",
            "motorcycle rear view", "bike rear view", "scooter rear view"
        ],
        "left_side_view": [
            "car left side view", "automobile left side", "vehicle left profile",
            "motorcycle left side", "bike left side view", "scooter left side"
        ],
        "right_side_view": [
            "car right side view", "automobile right side", "vehicle right profile",
            "motorcycle right side", "bike right side view", "scooter right side"
        ]
    }
    
    for orientation in orientations:
        print(f"\nDownloading images for {orientation}...")
        
        # Download images using multiple search terms for each orientation
        for search_term in search_terms[orientation]:
            # Limit the number of images per search term
            images_per_term = max(10, images_per_orientation // len(search_terms[orientation]))
            
            print(f"  Searching for: '{search_term}'")
            response.download(search_term, images_per_term)
            
            # Add a delay to avoid being blocked
            time.sleep(random.uniform(1.0, 3.0))
    
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
        # Get all search terms used for this orientation
        orientation_search_terms = search_terms[orientation]
        
        # Move images from each search term folder to the orientation folder
        for search_term in orientation_search_terms:
            # Directory name with spaces (as used by simple_image_download)
            search_term_dir = os.path.join(download_dir, search_term)
            if os.path.exists(search_term_dir):
                print(f"  Copying images from '{search_term}' to {orientation}...")
                
                for img_file in os.listdir(search_term_dir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        # Create a unique filename to avoid overwriting
                        src_path = os.path.join(search_term_dir, img_file)
                        dst_path = os.path.join(base_dir, orientation, f"{search_term.replace(' ', '_')}_{img_file}")
                        
                        # Copy the file
                        try:
                            shutil.copy2(src_path, dst_path)
                            total_copied += 1
                        except Exception as e:
                            print(f"Error copying {src_path}: {e}")
    
    print(f"Images organized successfully! Copied {total_copied} images.")
    return total_copied

def main():
    # Create directory structure
    base_dir, orientations = create_directories()
    
    # Download images
    search_terms = download_images(orientations, images_per_orientation=40)
    
    # Organize images into orientation folders
    total_copied = organize_images(base_dir, orientations, search_terms)
    
    print("\nDataset collection complete!")
    print(f"Your dataset is available in the '{base_dir}' directory")
    print(f"Total images in dataset: {total_copied}")

if __name__ == "__main__":
    main() 