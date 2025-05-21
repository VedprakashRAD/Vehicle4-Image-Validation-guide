#!/bin/bash

# Get the IP address of the host machine (works on macOS and Linux)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    HOST_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
else
    # Linux
    HOST_IP=$(hostname -I | awk '{print $1}')
fi

echo "======================================================="
echo "Vehicle Image Validation - Wireless Testing Setup"
echo "======================================================="
echo ""
echo "Host IP Address: $HOST_IP"
echo ""

# Make sure we are in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing processes on port 9000
echo "Checking for existing processes on port 9000..."
PID=$(lsof -ti:9000 || echo "")
if [ ! -z "$PID" ]; then
    echo "Killing existing process on port 9000 (PID: $PID)"
    kill -9 $PID
fi

# Start FastAPI backend using python directly (no docker)
echo "1. Starting the FastAPI backend..."
# Check if Python virtual environment exists
if [ -d ".venv" ]; then
    echo "   Using existing Python virtual environment"
    source .venv/bin/activate
else
    echo "   Creating new Python virtual environment"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Start the FastAPI backend in the background
echo "   Starting FastAPI server on port 9000..."
python -m uvicorn main:app --host 0.0.0.0 --port 9000 &
FASTAPI_PID=$!
echo "   Backend running with PID: $FASTAPI_PID"
sleep 2
echo ""

# Update API URL in the Flutter app
echo "2. Updating API URL in the Flutter app..."
sed -i '' "s|final String apiBaseUrl = 'http://localhost:9000';|final String apiBaseUrl = 'http://$HOST_IP:9000';|g" vehicle_image_validation/lib/screens/vehicle_image_uploader.dart
echo "   Updated vehicle_image_uploader.dart to use $HOST_IP"
echo ""

# Build web version
echo "3. Building the Flutter web app..."
cd vehicle_image_validation
flutter build web
echo ""

# Create a simple HTML server in the current directory
echo "4. Starting the web server..."
# First make sure we're in the right directory
if [ -d "build/web" ]; then
    cd build/web
else
    echo "Error: build/web directory not found!"
    echo "Creating a simple web server in the current directory instead."
fi

python3 -m http.server 8080 &
WEB_SERVER_PID=$!
echo ""

echo "5. ACCESS THE APP:"
echo "   Open a browser on your Android device and navigate to:"
echo "   http://$HOST_IP:8080"
echo ""
echo "6. Press Ctrl+C to stop all servers when finished..."
echo ""
echo "======================================================="

# Wait for user to press Ctrl+C and then clean up
trap "kill $WEB_SERVER_PID 2>/dev/null; kill $FASTAPI_PID 2>/dev/null; echo ''; echo 'All servers stopped.'; echo ''" INT
wait 