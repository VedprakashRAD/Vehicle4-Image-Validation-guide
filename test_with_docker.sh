#!/bin/bash

# Get the IP address of the host machine (works on macOS and Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    HOST_IP=$(ipconfig getifaddr en0)
else
    # Linux
    HOST_IP=$(hostname -I | awk '{print $1}')
fi

echo "======================================================="
echo "Setting up Vehicle Image Validation Test Environment"
echo "======================================================="
echo ""
echo "Host IP Address: $HOST_IP"
echo ""
echo "1. Starting the FastAPI backend with Docker..."
docker-compose up -d
echo ""
echo "2. Backend server is running at: http://$HOST_IP:9000"
echo ""
echo "3. To test in Flutter:"
echo "   - Update the apiBaseUrl in vehicle_image_validation/lib/screens/vehicle_image_uploader.dart to:"
echo "     http://$HOST_IP:9000"
echo ""
echo "4. Run the Flutter app on your device with:"
echo "   cd vehicle_image_validation && flutter run"
echo ""
echo "5. To stop the Docker container when finished:"
echo "   docker-compose down"
echo ""
echo "======================================================="
echo "If using an Android emulator, use 10.0.2.2 instead of your host IP."
echo "=======================================================" 