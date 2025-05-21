#!/bin/bash

# Get IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    HOST_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
else
    HOST_IP=$(hostname -I | awk '{print $1}')
fi

echo "======================================================="
echo "FIXED: Vehicle Image Validation Web Test"
echo "======================================================="
echo "Host IP Address: $HOST_IP"
echo ""

# Make sure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Fix 1: Force kill any process using port 9000
kill_port() {
    local port=$1
    echo "Checking for processes on port $port..."
    PID=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$PID" ]; then
        echo "Killing process on port $port (PID: $PID)"
        kill -9 $PID 2>/dev/null
        sleep 1
    else
        echo "No process found on port $port"
    fi
}

# Kill processes on ports we'll use
kill_port 9000
kill_port 8080

# Fix 2: Update API URL in the Flutter code before building
echo "1. Updating API URL to use $HOST_IP..."
cd vehicle_image_validation
echo "Current directory: $(pwd)"

# Update the API URL in the main screen
sed -i '' "s|final String apiBaseUrl = 'http://localhost:9000';|final String apiBaseUrl = 'http://$HOST_IP:9000';|g" lib/screens/vehicle_image_uploader.dart 2>/dev/null || echo "Warning: Could not update API URL in vehicle_image_uploader.dart"

# Fix 3: Clean and rebuild Flutter
echo "2. Cleaning and rebuilding the Flutter web app..."
flutter clean
flutter pub get
flutter build web

# Fix 4: Ensure the web directory exists
echo "3. Checking if build/web directory exists..."
if [ -d "build/web" ]; then
    echo "   build/web directory found"
    cd build/web
else
    echo "   build/web directory not found, creating simple HTML page"
    mkdir -p build/web
    cd build/web
    cat > index.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Vehicle Image Validation</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
        h1 { color: #3498db; }
        .container { max-width: 600px; margin: 0 auto; }
        .btn { background-color: #3498db; color: white; padding: 10px 20px; border: none; 
               border-radius: 4px; cursor: pointer; margin: 10px; }
        .mock-screen { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Vehicle Image Validation</h1>
        <p>Web Testing Mode</p>
        
        <div class="mock-screen">
            <h2>Welcome!</h2>
            <p>Your session ID: mock-123456</p>
            <button class="btn">START INSPECTION</button>
        </div>
        
        <div class="mock-screen">
            <h2>Select a side to capture:</h2>
            <button class="btn">FRONT</button>
            <button class="btn">REAR</button>
            <button class="btn">LEFT</button>
            <button class="btn">RIGHT</button>
        </div>
        
        <p>API Server: http://$HOST_IP:9000</p>
    </div>
</body>
</html>
EOF
fi

# Fix 5: Start the web server
echo "4. Starting web server on port 8080..."
python3 -m http.server 8080 &
WEB_SERVER_PID=$!

echo ""
echo "5. ACCESS THE APP:"
echo "   Open a browser on your Android device and navigate to:"
echo "   http://$HOST_IP:8080"
echo ""
echo "6. Press Ctrl+C to stop the server when finished..."
echo "======================================================="

# Wait for Ctrl+C
trap "kill $WEB_SERVER_PID 2>/dev/null; echo ''; echo 'Server stopped.'; echo ''" INT
wait 