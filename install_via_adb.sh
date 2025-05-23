#!/bin/bash

# Script to install the APK via ADB (Android Debug Bridge)
# This requires the Android device to be connected via USB and have USB debugging enabled

# Check if ADB is installed
if ! command -v adb &> /dev/null; then
    echo "Error: ADB (Android Debug Bridge) is not installed or not in PATH"
    echo "Please install Android SDK Platform Tools"
    exit 1
fi

# Check if APK file exists
APK_FILE="app-debug.apk"
if [ ! -f "$APK_FILE" ]; then
    echo "Error: $APK_FILE not found!"
    exit 1
fi

# Check for connected devices
DEVICES=$(adb devices | grep -v "List" | grep "device" | wc -l)
if [ "$DEVICES" -eq 0 ]; then
    echo "Error: No Android devices connected"
    echo "Please connect your device via USB and enable USB debugging"
    exit 1
fi

echo "Found $DEVICES connected device(s)"

# Install the APK
echo "Installing $APK_FILE..."
adb install -r "$APK_FILE"

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Installation successful!"
    echo ""
    echo "After opening the app, use this server address:"
    echo "http://10.1.1.181:9000"
else
    echo "Installation failed!"
fi 