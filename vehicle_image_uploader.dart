import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'vehicle_camera_guide.dart';
import 'package:camera/camera.dart';

enum VehicleSide { front, rear, left, right }

enum UploadStatus { pending, uploading, success, failed }

class VehicleImageUploader extends StatefulWidget {
  final String sessionId;
  
  const VehicleImageUploader({
    Key? key,
    required this.sessionId,
  }) : super(key: key);

  @override
  _VehicleImageUploaderState createState() => _VehicleImageUploaderState();
}

class _VehicleImageUploaderState extends State<VehicleImageUploader> {
  // Base URL can be configured for different environments
  // For local development: http://localhost:9000
  // For railway: https://your-app-name.up.railway.app
  // For production: https://your-production-domain.com
  final String apiBaseUrl = 'http://192.168.10.131:9000'; // Can be changed to railway URL
  
  // Support for railway deployment - add this new method
  String getApiBaseUrl() {
    // Check if URL is stored in SharedPreferences or passed via parameters
    // For now, return the default URL
    return apiBaseUrl;
  }
  
  Map<VehicleSide, UploadStatus> uploadStatus = {
    VehicleSide.front: UploadStatus.pending,
    VehicleSide.rear: UploadStatus.pending,
    VehicleSide.left: UploadStatus.pending,
    VehicleSide.right: UploadStatus.pending,
  };
  
  Map<VehicleSide, String?> imagePaths = {
    VehicleSide.front: null,
    VehicleSide.rear: null,
    VehicleSide.left: null,
    VehicleSide.right: null,
  };
  
  bool _isLoading = true;
  String _statusMessage = "Checking session status...";
  bool _isComplete = false;
  List<CameraDescription> _cameras = [];
  
  @override
  void initState() {
    super.initState();
    _initCameras();
    _checkSessionStatus();
  }
  
  Future<void> _initCameras() async {
    try {
      _cameras = await availableCameras();
    } catch (e) {
      print('Error initializing cameras: $e');
    }
  }
  
  Future<void> _checkSessionStatus() async {
    setState(() {
      _isLoading = true;
      _statusMessage = "Checking session status...";
    });
    
    try {
      final response = await http.get(
        Uri.parse('${getApiBaseUrl()}/session/${widget.sessionId}'),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _updateStatusFromResponse(data);
      } else {
        setState(() {
          _statusMessage = "Error checking session: ${response.statusCode}";
          _isLoading = false;
        });
      }
    } catch (e) {
      print('Error checking session: $e');
      setState(() {
        _statusMessage = "Network error. Please check your connection.";
        _isLoading = false;
      });
    }
  }
  
  void _updateStatusFromResponse(Map<String, dynamic> data) {
    final sides = data['vehicle_sides'] ?? {};
    
    setState(() {
      // Update upload status for each side
      if (sides.containsKey('front')) {
        uploadStatus[VehicleSide.front] = sides['front'] ? UploadStatus.success : UploadStatus.pending;
      }
      if (sides.containsKey('rear')) {
        uploadStatus[VehicleSide.rear] = sides['rear'] ? UploadStatus.success : UploadStatus.pending;
      }
      if (sides.containsKey('left')) {
        uploadStatus[VehicleSide.left] = sides['left'] ? UploadStatus.success : UploadStatus.pending;
      }
      if (sides.containsKey('right')) {
        uploadStatus[VehicleSide.right] = sides['right'] ? UploadStatus.success : UploadStatus.pending;
      }
      
      // Check if all sides are uploaded
      final allUploaded = uploadStatus.values.every((status) => status == UploadStatus.success);
      _isComplete = allUploaded;
      
      if (_isComplete) {
        _statusMessage = "All vehicle sides have been successfully captured and validated!";
      } else {
        _statusMessage = "Please capture all sides of your vehicle";
      }
      
      _isLoading = false;
    });
  }
  
  Future<void> _launchCameraGuide(VehicleSide side) async {
    if (_cameras.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('No camera available')),
      );
      return;
    }
    
    final String sideName = _getSideName(side);
    
    // Set status to uploading to show progress
    setState(() {
      uploadStatus[side] = UploadStatus.uploading;
    });
    
    try {
      final imagePath = await Navigator.push<String>(
        context,
        MaterialPageRoute(
          builder: (context) => VehicleCameraGuide(
            vehicleSide: sideName,
            cameras: _cameras,
          ),
        ),
      );
      
      if (imagePath != null) {
        setState(() {
          imagePaths[side] = imagePath;
        });
        
        // Upload the image
        await _uploadImage(side, imagePath);
      } else {
        // User cancelled
        setState(() {
          uploadStatus[side] = imagePaths[side] != null ? UploadStatus.success : UploadStatus.pending;
        });
      }
    } catch (e) {
      print('Error during camera flow: $e');
      setState(() {
        uploadStatus[side] = imagePaths[side] != null ? UploadStatus.success : UploadStatus.failed;
      });
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error capturing image: $e')),
      );
    }
  }
  
  Future<void> _uploadImage(VehicleSide side, String imagePath) async {
    setState(() {
      uploadStatus[side] = UploadStatus.uploading;
      _statusMessage = "Uploading ${_getSideName(side)} image...";
    });
    
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('${getApiBaseUrl()}/upload_image/${widget.sessionId}'),
      );
      
      request.files.add(
        await http.MultipartFile.fromPath(
          'image',
          imagePath,
        ),
      );
      
      request.fields['side'] = _getSideName(side);
      
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final isValid = data['is_valid'] ?? false;
        
        setState(() {
          if (isValid) {
            uploadStatus[side] = UploadStatus.success;
            _statusMessage = "Successfully uploaded ${_getSideName(side)} image!";
          } else {
            uploadStatus[side] = UploadStatus.failed;
            _statusMessage = data['message'] ?? "Invalid image: general error";
          }
        });
      } else {
        setState(() {
          uploadStatus[side] = UploadStatus.failed;
          _statusMessage = "Error uploading: ${response.statusCode}";
        });
      }
      
      // Check overall session status
      await _checkSessionStatus();
    } catch (e) {
      print('Error uploading image: $e');
      setState(() {
        uploadStatus[side] = UploadStatus.failed;
        _statusMessage = "Network error during upload";
      });
    }
  }
  
  String _getSideName(VehicleSide side) {
    switch (side) {
      case VehicleSide.front: return "front";
      case VehicleSide.rear: return "rear";
      case VehicleSide.left: return "left";
      case VehicleSide.right: return "right";
    }
  }
  
  void _completeSession() {
    // Return true to indicate complete session
    Navigator.of(context).pop(true);
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Vehicle Image Uploader'),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(false),
        ),
      ),
      body: _isLoading
          ? Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Status card
                    Card(
                      elevation: 3,
                      child: Padding(
                        padding: EdgeInsets.all(16.0),
                        child: Column(
                          children: [
                            Text(
                              'Session ID: ${widget.sessionId}',
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.grey[700],
                              ),
                            ),
                            SizedBox(height: 12),
                            Text(
                              _statusMessage,
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                              textAlign: TextAlign.center,
                            ),
                            SizedBox(height: 16),
                            LinearProgressIndicator(
                              value: _getCompletionPercentage() / 100,
                              backgroundColor: Colors.grey[200],
                            ),
                            SizedBox(height: 8),
                            Text(
                              '${_getCompletionPercentage()}% Complete',
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      ),
                    ),
                    
                    SizedBox(height: 24),
                    
                    // Instructions
                    Text(
                      'Capture all sides of your vehicle',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Tap on each button to capture the corresponding view. Our system will validate the image quality.',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[700],
                      ),
                      textAlign: TextAlign.center,
                    ),
                    
                    SizedBox(height: 24),
                    
                    // Vehicle image upload grid
                    GridView.count(
                      crossAxisCount: 2,
                      childAspectRatio: 1.0,
                      crossAxisSpacing: 16,
                      mainAxisSpacing: 16,
                      shrinkWrap: true,
                      physics: NeverScrollableScrollPhysics(),
                      children: [
                        _buildSideButton(VehicleSide.front, Icons.arrow_upward),
                        _buildSideButton(VehicleSide.rear, Icons.arrow_downward),
                        _buildSideButton(VehicleSide.left, Icons.arrow_back),
                        _buildSideButton(VehicleSide.right, Icons.arrow_forward),
                      ],
                    ),
                    
                    SizedBox(height: 24),
                    
                    // Complete button
                    if (_isComplete)
                      ElevatedButton(
                        onPressed: _completeSession,
                        child: Padding(
                          padding: EdgeInsets.all(12.0),
                          child: Text(
                            'COMPLETE INSPECTION',
                            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                          ),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8.0),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),
    );
  }
  
  Widget _buildSideButton(VehicleSide side, IconData directionIcon) {
    final Color backgroundColor;
    final Widget stateWidget;
    final bool isEnabled = uploadStatus[side] != UploadStatus.uploading;
    
    switch (uploadStatus[side]) {
      case UploadStatus.success:
        backgroundColor = Colors.green.withOpacity(0.2);
        stateWidget = Icon(Icons.check_circle, color: Colors.green, size: 24);
        break;
      case UploadStatus.uploading:
        backgroundColor = Colors.blue.withOpacity(0.2);
        stateWidget = SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(
            strokeWidth: 2.0,
            valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
          ),
        );
        break;
      case UploadStatus.failed:
        backgroundColor = Colors.red.withOpacity(0.2);
        stateWidget = Icon(Icons.error, color: Colors.red, size: 24);
        break;
      case UploadStatus.pending:
      default:
        backgroundColor = Colors.grey.withOpacity(0.1);
        stateWidget = Icon(Icons.camera_alt, color: Colors.blue, size: 24);
        break;
    }
    
    final hasImage = imagePaths[side] != null;
    
    return GestureDetector(
      onTap: isEnabled ? () => _launchCameraGuide(side) : null,
      child: Container(
        decoration: BoxDecoration(
          color: backgroundColor,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: uploadStatus[side] == UploadStatus.success
                ? Colors.green
                : uploadStatus[side] == UploadStatus.failed
                    ? Colors.red
                    : Colors.grey.withOpacity(0.3),
            width: 2,
          ),
        ),
        child: Stack(
          children: [
            // If we have an image, show it as background
            if (hasImage)
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: Opacity(
                  opacity: 0.6,
                  child: Image.file(
                    File(imagePaths[side]!),
                    fit: BoxFit.cover,
                    width: double.infinity,
                    height: double.infinity,
                  ),
                ),
              ),
            
            // Content
            Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Direction icon
                Icon(
                  directionIcon,
                  size: 36,
                  color: Colors.black54,
                ),
                
                SizedBox(height: 12),
                
                // Side name
                Text(
                  _getSideName(side).toUpperCase(),
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                
                SizedBox(height: 12),
                
                // Status indicator
                stateWidget,
              ],
            ),
          ],
        ),
      ),
    );
  }
  
  double _getCompletionPercentage() {
    int completed = 0;
    uploadStatus.forEach((side, status) {
      if (status == UploadStatus.success) {
        completed++;
      }
    });
    
    return (completed / uploadStatus.length) * 100;
  }
} 