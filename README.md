# Vehicle Orientation Detection System

This project implements a vehicle orientation detection system that can identify if uploaded vehicle images (cars, bikes, scooters) are correctly oriented as front, rear, left, or right views.

## Project Overview

The system can verify if an uploaded image matches the expected orientation and provide appropriate feedback messages like "Front side uploaded successfully" or "Mismatch: Left side is detected for the intended Right side".

## Components

1. **Data Collection**
   - `collect_large_dataset.py`: Downloads vehicle images from the web for four orientations (front, rear, left, right) including cars, bikes, and scooters.

2. **Data Preprocessing**
   - `preprocess_large_dataset.py`: Resizes, normalizes, and augments the collected images for better model training.
   - `clean_dataset.py`: Removes corrupted images from the dataset.

3. **Model Training**
   - `train_improved_model.py`: Trains an EfficientNetB0 model to classify vehicle orientations.

4. **Testing and Evaluation**
   - `test_orientation_model.py`: Tests the trained model on new images.
   - `download_test_images.py`: Downloads test images for evaluation.

5. **Integration**
   - `vehicle_orientation_verifier.py`: Demonstrates how to use the model in a real application.

## Usage

### Collecting Dataset

```bash
python collect_large_dataset.py
```

### Preprocessing Dataset

```bash
python preprocess_large_dataset.py --input_dir large_vehicle_dataset --output_dir processed_large_dataset
```

### Training Model

```bash
python train_improved_model.py
```

### Testing Model

```bash
python test_orientation_model.py --model [model_path] --test_dir test_images --output_dir test_results
```

### Verifying Vehicle Orientation

```bash
python vehicle_orientation_verifier.py --image [image_path] --expected [front/rear/left/right] --output [output_path]
```

## Model Performance

The model was trained on approximately 1,800 images across four orientations (front, rear, left, right). The current model has a bias towards detecting left-side views, which can be improved with a larger and more balanced dataset.

## Future Improvements

1. Collect a larger and more balanced dataset with 1,000+ images per orientation.
2. Implement more sophisticated data augmentation techniques.
3. Try different model architectures or ensemble methods.
4. Add more vehicle types and variations for better generalization.

## License

MIT License

## Credits

- YOLOv8: https://github.com/ultralytics/ultralytics
- TensorFlow: https://www.tensorflow.org/
- FastAPI: https://fastapi.tiangolo.com/ 

## Deployment on Railway

This application is configured for deployment on [Railway](https://railway.app). To deploy:

1. **Sign up for Railway**:
   - Create an account at [Railway.app](https://railway.app)
   - Install the Railway CLI if you want to deploy from your local machine

2. **Deploy the API**:
   - From the Railway dashboard, create a new project
   - Choose "Deploy from GitHub" and select your repository
   - Railway will automatically detect the configuration from `railway.toml`
   - The application will be deployed and a public URL will be provided

3. **Environment Variables**:
   - You may need to set environment variables in the Railway dashboard
   - Important variables include `PORT`, `MIN_WIDTH`, `MIN_HEIGHT`, etc.

4. **Update the Flutter App**:
   - Open the Flutter app code
   - Update the `apiBaseUrl` in `vehicle_image_uploader.dart` to your Railway URL:
     ```dart
     final String apiBaseUrl = 'https://your-railway-app.up.railway.app';
     ```
   - Rebuild and deploy the Flutter app

5. **Continuous Deployment**:
   - Railway automatically deploys when you push changes to your GitHub repository
   - You can configure specific branches for deployment in the Railway dashboard

6. **Custom Domain**:
   - In the Railway dashboard, go to Settings
   - Configure a custom domain if desired

For more details, refer to [Railway documentation](https://docs.railway.app). 