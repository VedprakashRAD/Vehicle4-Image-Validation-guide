# Vehicle Image Validation App

This Flutter application works with a FastAPI backend to help users capture and validate vehicle images from all sides (front, rear, left, right). The app guides users through the image capture process and validates images in real-time.

## Application Overview

The Flutter application has been successfully created with all the necessary components for vehicle image validation. The app includes:
- A main screen that manages sessions and provides access to the vehicle image uploader
- A dedicated vehicle image uploader screen with a grid layout for capturing all four sides of a vehicle
- A camera guide component with visual overlays to help users properly frame their vehicle during image capture
- Integration with a FastAPI backend through HTTP requests
- Proper permissions set up for both Android and iOS platforms

The application follows a user-friendly flow where:
- A session is created when the app starts
- Users can capture images of each vehicle side with guided assistance
- Images are validated in real-time with the backend
- Progress is tracked and users can complete the inspection when all sides are validated

To run the application, you'll need to:
- Have Flutter installed and set up on your development machine
- Run `flutter pub get` to install dependencies
- Connect to your FastAPI backend by updating the `apiBaseUrl` in `vehicle_image_uploader.dart`
- Run the app with `flutter run`

The app is now ready to be tested and integrated with your FastAPI backend!

## Features

- Camera guide with visual overlays to help position the vehicle correctly
- Real-time image validation
- Status tracking for each vehicle side
- Session management
- User-friendly interface with progress tracking
- Smart image framing guidance with vehicle-specific overlays
- Detailed feedback for image quality improvement
- Secure image uploads with session management

## Getting Started

### Prerequisites

- Flutter SDK (>= 2.17.0)
- Dart SDK
- Android Studio or VS Code with Flutter extensions
- An iOS or Android device (or emulator) for testing

### Installation

1. Clone this repository:
```
git clone <repository-url>
```

2. Navigate to the project directory:
```
cd vehicle_image_validation
```

3. Install dependencies:
```
flutter pub get
```

4. Update the FastAPI server URL:
   - Open `lib/vehicle_image_uploader.dart`
   - Update the `apiBaseUrl` variable with your FastAPI server URL

### Running the App

```
flutter run
```

## User Flow

1. **Home Screen**: The app starts with a home screen that creates a new session.
2. **Image Capturing**: Tapping "START INSPECTION" takes you to the capturing interface with four buttons for different vehicle sides.
3. **Guided Camera View**:
   - Each side has specific guidelines to help position the vehicle properly
   - Front and rear sides have a license plate detection area
   - Side views have horizontal alignment guides
4. **Image Validation**: After capturing, preview the image and choose to accept or retake.
5. **Upload and Verification**: The image is uploaded to the backend which validates if it's a proper vehicle image from the claimed angle.
6. **Session Completion**: Once all sides are captured, the app allows completion of the inspection.

## API Endpoints Required

The application expects the following API endpoints to be available on your FastAPI server:

1. `POST /session` - Creates a new session 
   - Returns: `{"session_id": "unique_id"}`

2. `GET /session/{session_id}` - Gets the status of a session
   - Returns: `{"vehicle_sides": {"front": bool, "rear": bool, "left": bool, "right": bool}}`

3. `POST /upload_image/{session_id}` - Uploads and validates an image
   - Form data:
     - `image`: Image file
     - `side`: String ("front", "rear", "left", or "right")
   - Returns: `{"is_valid": bool, "message": "validation message"}`

## Architecture

The app follows a simple yet effective architecture:

- `main.dart` - Entry point with app configuration and home screen
- `vehicle_image_uploader.dart` - Main upload interface for all vehicle sides
- `vehicle_camera_guide.dart` - Camera interface with visual guides for capture

### Key Components

#### Home Screen (`main.dart`)
- Manages session creation with FastAPI backend
- Provides user interface for starting/restarting the inspection process
- Handles session state and user feedback

#### Image Uploader (`vehicle_image_uploader.dart`)
- Grid interface for selecting which side of the vehicle to capture
- Tracks upload status for each side (pending, uploading, success, failed)  
- Shows previews of uploaded images and session progress
- Communicates with backend API to validate uploads

#### Camera Guide (`vehicle_camera_guide.dart`)
- Implements a camera view with specialized overlays for each vehicle side
- Provides visual guidance for proper vehicle positioning
- Offers image preview and accept/retake functionality
- Implements custom painters for the guide overlays specific to each vehicle side

### Technical Implementation

- **State Management**: Uses Flutter's StatefulWidget pattern for component-level state
- **Camera Access**: Implements the camera plugin with proper lifecycle management
- **Custom Drawing**: Uses CustomPainter for vehicle framing guidelines
- **API Integration**: HTTP-based communication with FastAPI backend
- **Image Handling**: Temporary storage and uploading of captured images

## Permissions

The app requires the following permissions:

### Android
- Camera (for capturing images)
- Internet (for API communication)
- Storage (for saving temporary images)

### iOS
- Camera usage
- Photo library usage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

- If camera permissions are denied, make sure to grant them in your device settings
- If images fail to upload, check your server URL and network connection
- iOS simulators don't support camera functionality - use a physical device for testing
- If the app crashes when accessing the camera, ensure you've added the proper permissions to both Android and iOS config files

## Future Enhancements

- Machine learning for automatic vehicle position detection
- Automatic image enhancement for poor lighting conditions
- Image blur detection and automatic notification
- Support for different vehicle types with specialized overlays
- Offline mode with upload queuing 