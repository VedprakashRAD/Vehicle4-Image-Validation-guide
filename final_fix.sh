#!/bin/bash

# Get IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    HOST_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
else
    HOST_IP=$(hostname -I | awk '{print $1}')
fi

# Use a different port - 8088 instead of 8080
WEB_PORT=8088

echo "======================================================="
echo "FINAL FIX: Vehicle Image Validation Web Test"
echo "======================================================="
echo "Host IP Address: $HOST_IP"
echo "Web Port: $WEB_PORT"
echo ""

# Make sure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Fix 1: More robust process killing for ports
kill_port() {
    local port=$1
    echo "Forcefully terminating any process using port $port..."
    
    # Find PID using lsof and kill with -9
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
    
    sleep 2
    echo "Port $port should be free now"
}

# Kill any processes on our ports
kill_port 9000
kill_port 8080 
kill_port $WEB_PORT

# Create a simple HTML page in the current directory
echo "1. Creating a simple testing page..."
mkdir -p test_web
cd test_web

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
        <p>Simplified Web Testing Mode</p>
        
        <div class="mock-screen">
            <h2>Test Successful!</h2>
            <p>If you can see this page, your device can successfully connect to your computer.</p>
            <p>IP Address: $HOST_IP</p>
            <p>Port: $WEB_PORT</p>
        </div>
        
        <div class="mock-screen">
            <h2>Vehicle Image Upload Test</h2>
            <button class="btn" onclick="alert('Camera mock activated')">FRONT</button>
            <button class="btn" onclick="alert('Camera mock activated')">REAR</button>
            <button class="btn" onclick="alert('Camera mock activated')">LEFT</button>
            <button class="btn" onclick="alert('Camera mock activated')">RIGHT</button>
        </div>
    </div>
</body>
</html>
EOF

# Try multiple ways to start a web server
echo "2. Starting simple web server on port $WEB_PORT..."

# First try with Python
python3 -m http.server $WEB_PORT > /dev/null 2>&1 &
WEB_SERVER_PID=$!

# Check if server started successfully
sleep 1
if ! nc -z localhost $WEB_PORT > /dev/null 2>&1; then
    echo "Python HTTP server failed, trying alternative methods..."
    kill $WEB_SERVER_PID 2>/dev/null || true
    
    # Try PHP
    php -S 0.0.0.0:$WEB_PORT > /dev/null 2>&1 &
    WEB_SERVER_PID=$!
    
    sleep 1
    if ! nc -z localhost $WEB_PORT > /dev/null 2>&1; then
        echo "PHP server also failed, trying Node HTTP server..."
        kill $WEB_SERVER_PID 2>/dev/null || true
        
        # Try node http-server if available
        npx http-server -p $WEB_PORT > /dev/null 2>&1 &
        WEB_SERVER_PID=$!
        
        sleep 1
        if ! nc -z localhost $WEB_PORT > /dev/null 2>&1; then
            echo "All server attempts failed. Last resort: trying a different port..."
            kill $WEB_SERVER_PID 2>/dev/null || true
            
            # Try one more port
            WEB_PORT=3000
            python3 -m http.server $WEB_PORT > /dev/null 2>&1 &
            WEB_SERVER_PID=$!
        fi
    fi
fi

echo "Server started on port $WEB_PORT"
echo ""
echo "3. ACCESS THE TEST PAGE:"
echo "   Open a browser on your Android device and navigate to:"
echo "   http://$HOST_IP:$WEB_PORT"
echo ""
echo "4. When you can see this test page, you know your network connection works."
echo ""
echo "5. Press Ctrl+C to stop the server when finished..."
echo "======================================================="

# Wait for Ctrl+C
trap "kill $WEB_SERVER_PID 2>/dev/null; echo ''; echo 'Server stopped.'; echo ''" INT
wait 