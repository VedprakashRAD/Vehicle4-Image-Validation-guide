#!/bin/bash

# Get the IP address of the host machine
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    HOST_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
else
    # Linux
    HOST_IP=$(hostname -I | awk '{print $1}')
fi

echo "======================================================="
echo "Simple Web Test for Vehicle Image Validation App"
echo "======================================================="
echo ""
echo "Host IP Address: $HOST_IP"
echo ""

# Make sure we are in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Go to the vehicle_image_validation directory
cd vehicle_image_validation

# Build the web version
echo "1. Building the Flutter web app..."
flutter build web
echo ""

# Serve the files
echo "2. Starting web server on port 8080..."
cd build/web
python3 -m http.server 8080 &
WEB_SERVER_PID=$!
echo ""

echo "3. ACCESS THE APP:"
echo "   Open a browser on your Android device and navigate to:"
echo "   http://$HOST_IP:8080"
echo ""
echo "4. Press Ctrl+C to stop the server when finished..."
echo ""
echo "======================================================="

# Wait for Ctrl+C
trap "kill $WEB_SERVER_PID 2>/dev/null; echo ''; echo 'Web server stopped.'; echo ''" INT
wait 