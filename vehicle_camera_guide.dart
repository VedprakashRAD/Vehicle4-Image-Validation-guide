import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:path/path.dart' as path;

// Enum for vehicle sides
enum VehicleSide { front, rear, left, right }

// Enum for feedback status
enum FeedbackStatus { 
  waitingForVehicle,    // No vehicle detected yet 
  vehicleDetected,      // Vehicle detected but not in position
  improperAngle,        // Vehicle detected but wrong angle
  tooClose,             // Vehicle too close
  tooFar,               // Vehicle too far away
  partiallyVisible,     // Not all edges visible
  poorLighting,         // Image too dark/bright
  blurry,               // Image is blurry
  perfect,              // Everything good, ready to capture
}

class VehicleCameraGuide extends StatefulWidget {
  final String vehicleSide;
  final List<CameraDescription> cameras;
  
  const VehicleCameraGuide({
    Key? key, 
    required this.vehicleSide,
    required this.cameras,
  }) : super(key: key);

  @override
  _VehicleCameraGuideState createState() => _VehicleCameraGuideState();
}

class _VehicleCameraGuideState extends State<VehicleCameraGuide> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool _isTakingPicture = false;
  String? _imagePath;
  
  @override
  void initState() {
    super.initState();
    // Initialize camera controller
    _controller = CameraController(
      // Use the first available camera (usually back camera)
      widget.cameras.first,
      // Use high resolution
      ResolutionPreset.high,
      enableAudio: false,
    );
    
    // Initialize controller future
    _initializeControllerFuture = _controller.initialize();
  }
  
  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed
    _controller.dispose();
    super.dispose();
  }
  
  Future<void> _takePicture() async {
    if (_isTakingPicture) return;
    
    setState(() {
      _isTakingPicture = true;
    });
    
    try {
      // Ensure the camera is initialized
      await _initializeControllerFuture;
      
      // Create a path where the image should be saved
      final directory = await getTemporaryDirectory();
      final imagePath = path.join(directory.path, '${widget.vehicleSide}_${DateTime.now().millisecondsSinceEpoch}.jpg');
      
      // Take the picture
      final XFile image = await _controller.takePicture();
      
      // Save the picture to our path
      await image.saveTo(imagePath);
      
      setState(() {
        _imagePath = imagePath;
        _isTakingPicture = false;
      });
    } catch (e) {
      print('Error taking picture: $e');
      setState(() {
        _isTakingPicture = false;
      });
    }
  }
  
  void _acceptImage() {
    if (_imagePath != null) {
      Navigator.of(context).pop(_imagePath);
    }
  }
  
  void _retakeImage() {
    setState(() {
      _imagePath = null;
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Capture ${_getVehicleSideLabel()}'),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: _imagePath == null ? _buildCameraView() : _buildPreviewView(),
    );
  }
  
  Widget _buildCameraView() {
    return FutureBuilder<void>(
      future: _initializeControllerFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          return Stack(
            fit: StackFit.expand,
            children: [
              // Camera preview
              CameraPreview(_controller),
              
              // Overlay with guide frame
              _buildGuideOverlay(),
              
              // Capture button at the bottom
              Positioned(
                bottom: 30,
                left: 0,
                right: 0,
                child: Center(
                  child: FloatingActionButton(
                    onPressed: _isTakingPicture ? null : _takePicture,
                    child: _isTakingPicture 
                        ? CircularProgressIndicator(color: Colors.white)
                        : Icon(Icons.camera_alt, size: 36),
                    backgroundColor: Colors.blue,
                  ),
                ),
              ),
              
              // Guide text
              Positioned(
                top: 20,
                left: 0,
                right: 0,
                child: Container(
                  padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  color: Colors.black54,
                  child: Text(
                    _getGuidanceText(),
                    style: TextStyle(color: Colors.white, fontSize: 16),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            ],
          );
        } else {
          return Center(child: CircularProgressIndicator());
        }
      },
    );
  }
  
  Widget _buildPreviewView() {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Show captured image
        Image.file(
          File(_imagePath!),
          fit: BoxFit.contain,
        ),
        
        // Action buttons at the bottom
        Positioned(
          bottom: 30,
          left: 0,
          right: 0,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton.icon(
                onPressed: _retakeImage,
                icon: Icon(Icons.refresh),
                label: Text('RETAKE'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                ),
              ),
              ElevatedButton.icon(
                onPressed: _acceptImage,
                icon: Icon(Icons.check),
                label: Text('ACCEPT'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
  
  Widget _buildGuideOverlay() {
    return CustomPaint(
      painter: GuideFramePainter(vehicleSide: widget.vehicleSide),
      child: Container(),
    );
  }
  
  String _getVehicleSideLabel() {
    switch (widget.vehicleSide.toLowerCase()) {
      case 'front':
        return 'Front of Vehicle';
      case 'rear':
        return 'Rear of Vehicle';
      case 'left':
        return 'Left Side of Vehicle';
      case 'right':
        return 'Right Side of Vehicle';
      default:
        return 'Vehicle Image';
    }
  }
  
  String _getGuidanceText() {
    switch (widget.vehicleSide.toLowerCase()) {
      case 'front':
        return 'Position the front of the vehicle within the frame. Ensure license plate is visible.';
      case 'rear':
        return 'Position the rear of the vehicle within the frame. Ensure license plate is visible.';
      case 'left':
        return 'Capture the entire left side of the vehicle within the frame.';
      case 'right':
        return 'Capture the entire right side of the vehicle within the frame.';
      default:
        return 'Position the vehicle properly within the frame.';
    }
  }
}

class GuideFramePainter extends CustomPainter {
  final String vehicleSide;
  
  GuideFramePainter({required this.vehicleSide});
  
  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = Colors.white.withOpacity(0.8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;
    
    final double frameMargin = 40.0;
    
    // Draw rectangle frame with rounded corners
    final Rect rect = Rect.fromLTRB(
      frameMargin, 
      size.height * 0.2, 
      size.width - frameMargin, 
      size.height * 0.8
    );
    
    final RRect rRect = RRect.fromRectAndRadius(
      rect, 
      Radius.circular(20.0),
    );
    
    canvas.drawRRect(rRect, paint);
    
    // Draw specific guides based on vehicle side
    if (vehicleSide.toLowerCase() == 'front' || vehicleSide.toLowerCase() == 'rear') {
      // Horizontal center line for alignment
      canvas.drawLine(
        Offset(size.width / 2, size.height * 0.2),
        Offset(size.width / 2, size.height * 0.8),
        paint..color = Colors.yellow.withOpacity(0.6),
      );
      
      // License plate area
      final Rect licensePlateRect = Rect.fromCenter(
        center: Offset(size.width / 2, size.height * 0.6),
        width: size.width * 0.3,
        height: size.height * 0.1,
      );
      
      canvas.drawRect(
        licensePlateRect,
        paint..color = Colors.blue.withOpacity(0.5),
      );
    } else if (vehicleSide.toLowerCase() == 'left' || vehicleSide.toLowerCase() == 'right') {
      // Horizontal guidelines for car profile
      canvas.drawLine(
        Offset(frameMargin, size.height * 0.5),
        Offset(size.width - frameMargin, size.height * 0.5),
        paint..color = Colors.yellow.withOpacity(0.6),
      );
    }
  }
  
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
} 