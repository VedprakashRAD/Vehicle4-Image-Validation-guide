import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_sample_image(text, output_path, width=800, height=600, bg_color=(240, 240, 240), text_color=(0, 0, 0)):
    """Create a sample image with text for demonstration purposes"""
    # Create a blank image with background color
    image = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (400, 40)
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Draw the text
    draw.text(position, text, fill=text_color, font=font)
    
    # Add a vehicle outline
    if "front" in text.lower():
        draw_front_vehicle(draw, width, height)
    elif "rear" in text.lower():
        draw_rear_vehicle(draw, width, height)
    elif "left" in text.lower():
        draw_left_vehicle(draw, width, height)
    elif "right" in text.lower():
        draw_right_vehicle(draw, width, height)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Created sample image: {output_path}")

def draw_front_vehicle(draw, width, height):
    """Draw a simplified front view of a vehicle"""
    # Car body - front view
    center_x = width // 2
    top_y = height // 2 - 100
    draw.rectangle([center_x - 150, top_y, center_x + 150, top_y + 100], outline=(50, 50, 50), width=3)
    
    # Headlights
    draw.ellipse([center_x - 120, top_y + 20, center_x - 80, top_y + 50], fill=(255, 255, 200), outline=(0, 0, 0))
    draw.ellipse([center_x + 80, top_y + 20, center_x + 120, top_y + 50], fill=(255, 255, 200), outline=(0, 0, 0))
    
    # Grille
    draw.rectangle([center_x - 60, top_y + 60, center_x + 60, top_y + 90], outline=(0, 0, 0))
    
    # Wheels
    draw.ellipse([center_x - 140, top_y + 90, center_x - 100, top_y + 130], fill=(50, 50, 50))
    draw.ellipse([center_x + 100, top_y + 90, center_x + 140, top_y + 130], fill=(50, 50, 50))

def draw_rear_vehicle(draw, width, height):
    """Draw a simplified rear view of a vehicle"""
    # Car body - rear view
    center_x = width // 2
    top_y = height // 2 - 100
    draw.rectangle([center_x - 150, top_y, center_x + 150, top_y + 100], outline=(50, 50, 50), width=3)
    
    # Taillights
    draw.rectangle([center_x - 120, top_y + 20, center_x - 80, top_y + 40], fill=(255, 0, 0), outline=(0, 0, 0))
    draw.rectangle([center_x + 80, top_y + 20, center_x + 120, top_y + 40], fill=(255, 0, 0), outline=(0, 0, 0))
    
    # License plate area
    draw.rectangle([center_x - 40, top_y + 60, center_x + 40, top_y + 80], fill=(200, 200, 200), outline=(0, 0, 0))
    
    # Wheels
    draw.ellipse([center_x - 140, top_y + 90, center_x - 100, top_y + 130], fill=(50, 50, 50))
    draw.ellipse([center_x + 100, top_y + 90, center_x + 140, top_y + 130], fill=(50, 50, 50))

def draw_left_vehicle(draw, width, height):
    """Draw a simplified left side view of a vehicle"""
    # Car body - side view
    center_x = width // 2
    center_y = height // 2
    draw.rectangle([center_x - 150, center_y - 50, center_x + 150, center_y + 30], outline=(50, 50, 50), width=3)
    
    # Car roof
    draw.polygon([
        (center_x - 50, center_y - 50),
        (center_x + 80, center_y - 50),
        (center_x + 120, center_y - 20),
        (center_x - 80, center_y - 20)
    ], outline=(50, 50, 50))
    
    # Windows
    draw.rectangle([center_x - 40, center_y - 40, center_x + 70, center_y - 25], fill=(200, 255, 255))
    
    # Wheels
    draw.ellipse([center_x - 100, center_y + 15, center_x - 60, center_y + 55], fill=(50, 50, 50))
    draw.ellipse([center_x + 80, center_y + 15, center_x + 120, center_y + 55], fill=(50, 50, 50))

def draw_right_vehicle(draw, width, height):
    """Draw a simplified right side view of a vehicle"""
    # Car body - side view (mirrored from left)
    center_x = width // 2
    center_y = height // 2
    draw.rectangle([center_x - 150, center_y - 50, center_x + 150, center_y + 30], outline=(50, 50, 50), width=3)
    
    # Car roof
    draw.polygon([
        (center_x + 50, center_y - 50),
        (center_x - 80, center_y - 50),
        (center_x - 120, center_y - 20),
        (center_x + 80, center_y - 20)
    ], outline=(50, 50, 50))
    
    # Windows - fix coordinates order (x1 must be greater than x0)
    draw.rectangle([center_x - 70, center_y - 40, center_x + 40, center_y - 25], fill=(200, 255, 255))
    
    # Wheels
    draw.ellipse([center_x + 60, center_y + 15, center_x + 100, center_y + 55], fill=(50, 50, 50))
    draw.ellipse([center_x - 120, center_y + 15, center_x - 80, center_y + 55], fill=(50, 50, 50))

def main():
    """Create sample images for all vehicle sides"""
    base_dir = Path("sample_images")
    
    # Create sample images for each side
    create_sample_image("FRONT VIEW EXAMPLE", base_dir / "front" / "sample_front.jpg")
    create_sample_image("REAR VIEW EXAMPLE", base_dir / "rear" / "sample_rear.jpg")
    create_sample_image("LEFT SIDE VIEW EXAMPLE", base_dir / "left" / "sample_left.jpg")
    create_sample_image("RIGHT SIDE VIEW EXAMPLE", base_dir / "right" / "sample_right.jpg")
    
    print("Sample images created successfully!")

if __name__ == "__main__":
    main() 