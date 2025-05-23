import os
import argparse
import time
from vehicle_orientation_verifier import load_orientation_model, verify_orientation, visualize_verification

def simulate_vehicle_registration():
    """Simulate a vehicle registration process with orientation verification"""
    print("\n===== Vehicle Registration System =====")
    print("This demo simulates uploading vehicle images from different angles\n")
    
    # Load the orientation model
    model_path = "orientation_model_efficientnet_20250523-172916/best_model.h5"
    model = load_orientation_model(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Create output directory for verification results
    output_dir = "registration_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Required orientations for registration
    required_orientations = ["front", "rear", "left", "right"]
    registration_status = {orientation: False for orientation in required_orientations}
    
    # Simulate the registration process
    while not all(registration_status.values()):
        print("\nCurrent registration status:")
        for orientation, status in registration_status.items():
            status_text = "✓ Uploaded" if status else "✗ Missing"
            print(f"  {orientation.capitalize()} view: {status_text}")
        
        # Ask which orientation to upload
        missing_orientations = [o for o, status in registration_status.items() if not status]
        print("\nPlease upload the following views:", ", ".join(missing_orientations))
        
        orientation = input("\nWhich view would you like to upload? (front/rear/left/right): ").lower()
        if orientation not in required_orientations:
            print("Invalid orientation. Please choose from: front, rear, left, right")
            continue
        
        # Ask for image path
        image_path = input(f"Enter the path to the {orientation} view image: ")
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            continue
        
        print(f"\nVerifying {orientation} view image...")
        time.sleep(1)  # Simulate processing time
        
        # Verify the orientation
        result = verify_orientation(model, image_path, orientation)
        
        # Display verification result
        print("\nVerification Result:")
        print(f"Valid: {result['is_valid']}")
        print(f"Message: {result['message']}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']:.4f}")
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{orientation}_verification.jpg")
        visualize_verification(image_path, result, output_path)
        print(f"Visualization saved to {output_path}")
        
        # Update registration status if valid
        if result["is_valid"]:
            registration_status[orientation] = True
            print(f"\n✓ {orientation.capitalize()} view accepted!")
        else:
            print(f"\n✗ {orientation.capitalize()} view rejected. Please try again.")
    
    # Registration complete
    print("\n===== Registration Complete! =====")
    print("All required vehicle views have been successfully uploaded and verified.")
    print(f"Verification results saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    simulate_vehicle_registration() 