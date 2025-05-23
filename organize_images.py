import os
import shutil
import glob

# Define source and destination directories
source_dir = "simple_images"
dest_dir = "dataset"

# Create mapping for orientation
right_keywords = [
    "right side", "right angle", "right profile", "right side view", "passenger side"
]

left_keywords = [
    "left side", "left angle", "left profile", "left side view", "driver side"
]

# Count variables
copied_to_right = 0
copied_to_left = 0
copied_to_front = 0
copied_to_rear = 0
errors = 0

# Process all subdirectories in simple_images
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    
    if not os.path.isdir(subdir_path):
        continue
    
    # Determine the target directory based on folder name
    target_dir = None
    
    # Check if it's a right-side image
    if any(keyword in subdir.lower() for keyword in right_keywords):
        target_dir = os.path.join(dest_dir, "right")
        copied_to_right += 1
    
    # Check if it's a left-side image
    elif any(keyword in subdir.lower() for keyword in left_keywords):
        target_dir = os.path.join(dest_dir, "left")
        copied_to_left += 1
    
    # If we couldn't determine the orientation, skip this directory
    if target_dir is None:
        print(f"Skipping {subdir} - couldn't determine orientation")
        continue
    
    # Copy all files from this subdirectory to the target directory
    try:
        for file_path in glob.glob(os.path.join(subdir_path, "*")):
            if os.path.isfile(file_path):
                shutil.copy2(file_path, target_dir)
    except Exception as e:
        print(f"Error copying files from {subdir_path}: {e}")
        errors += 1

print(f"Copied {copied_to_right} directories to right")
print(f"Copied {copied_to_left} directories to left")
print(f"Copied {copied_to_front} directories to front")
print(f"Copied {copied_to_rear} directories to rear")
print(f"Encountered {errors} errors") 