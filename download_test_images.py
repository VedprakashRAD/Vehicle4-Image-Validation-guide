from simple_image_download import simple_image_download as simp
import os
import time
import random
import shutil

def download_test_images(num_images_per_orientation=2):
    """Download test images for each orientation"""
    print(f"Downloading {num_images_per_orientation} test images for each orientation...")
    
    response = simp.simple_image_download()
    
    # Define search terms for each orientation
    search_terms = {
        "front_view": ["car front photo", "motorcycle front photo", "scooter front photo"],
        "rear_view": ["car rear photo", "motorcycle rear photo", "scooter rear photo"],
        "left_side_view": ["car left side photo", "motorcycle left side photo", "scooter left side photo"],
        "right_side_view": ["car right side photo", "motorcycle right side photo", "scooter right side photo"]
    }
    
    # Create test_images directory if it doesn't exist
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Download images for each orientation
    for orientation, terms in search_terms.items():
        print(f"\nDownloading test images for {orientation}...")
        
        for search_term in terms:
            # Download images
            print(f"  Searching for: '{search_term}'")
            response.download(search_term, num_images_per_orientation)
            
            # Add a delay to avoid being blocked
            time.sleep(random.uniform(1.0, 2.0))
    
    # Copy images to test_images directory
    print("\nCopying images to test_images directory...")
    download_dir = "simple_images"
    
    total_copied = 0
    
    for orientation, terms in search_terms.items():
        for search_term in terms:
            search_term_dir = os.path.join(download_dir, search_term)
            if os.path.exists(search_term_dir):
                for i, img_file in enumerate(os.listdir(search_term_dir)[:num_images_per_orientation]):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(search_term_dir, img_file)
                        dst_path = os.path.join(test_dir, f"test_{orientation}_{search_term.replace(' ', '_')}_{i+1}.jpg")
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                            total_copied += 1
                        except Exception as e:
                            print(f"Error copying {src_path}: {e}")
    
    print(f"Downloaded and copied {total_copied} test images to {test_dir} directory.")

if __name__ == "__main__":
    download_test_images(num_images_per_orientation=2) 