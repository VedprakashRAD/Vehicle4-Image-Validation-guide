# Vehicle Orientation Dataset Collection and Training Pipeline

This project provides a complete pipeline for collecting, processing, and training a vehicle orientation classification model. The model can accurately identify the orientation of vehicles (front, rear, left, right, etc.) in images, which is useful for the vehicle identification system described in the problem statement.

## Features

- **Dataset Collection**: Automatically collects vehicle images from the web for different orientations
- **Data Preprocessing**: Resizes, normalizes, and augments images for better model training
- **Model Training**: Trains a deep learning model (ResNet50 or MobileNetV2) for vehicle orientation classification
- **Model Testing**: Tests the trained model on new images and visualizes results

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- Pillow
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Simple-image-download

All dependencies can be installed using:

```bash
pip install -r requirements_orientation.txt
```

## Project Structure

```
.
├── collect_orientation_dataset.py   # Script to collect vehicle images from the web
├── preprocess_dataset.py            # Script to preprocess and augment the dataset
├── train_orientation_model.py       # Script to train the orientation classification model
├── test_orientation_model.py        # Script to test the model on new images
├── run_pipeline.sh                  # Shell script to run the entire pipeline
├── requirements_orientation.txt     # Required packages for the pipeline
└── vehicle_orientation_README.md    # This README file
```

## How to Run

1. Install the required packages:

```bash
pip install -r requirements_orientation.txt
```

2. Run the entire pipeline:

```bash
./run_pipeline.sh
```

Or run each step individually:

```bash
# Step 1: Collect dataset
python collect_orientation_dataset.py

# Step 2: Preprocess dataset
python preprocess_dataset.py

# Step 3: Train model
python train_orientation_model.py

# Step 4: Test model
python test_orientation_model.py
```

## Output Directories

- `vehicle_orientation_dataset/`: Raw collected images
- `processed_dataset/`: Preprocessed and augmented images
- `orientation_model_resnet50/`: Trained model and evaluation results
- `test_images/`: Directory for test images
- `test_results/`: Visualization of model predictions

## Model Details

The default model architecture is ResNet50 with custom classification layers. The model is trained in two phases:
1. Initial training with frozen base layers
2. Fine-tuning with unfrozen layers

You can modify the model architecture, training parameters, and other settings in `train_orientation_model.py`.

## Integration with Vehicle Identification System

This orientation detection model can be integrated into the main vehicle identification system to:

1. Verify that vehicle images are uploaded in the correct orientation
2. Provide feedback when a mismatch is detected (e.g., "Mismatch: Front side is detected for the intended Rear Side")
3. Reject uploads where no vehicle is detected

## License

This project is licensed under the MIT License - see the LICENSE file for details. 