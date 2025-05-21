import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class VehicleImageUploader extends StatefulWidget {
  const VehicleImageUploader({Key? key}) : super(key: key);

  @override
  _VehicleImageUploaderState createState() => _VehicleImageUploaderState();
}

class _VehicleImageUploaderState extends State<VehicleImageUploader> {
  final String apiBaseUrl = 'https://your-railway-app-url.com'; // Replace with your actual Railway URL
  String? sessionId;
  final Map<String, bool> uploadedSides = {
    'front': false,
    'rear': false,
    'left': false,
    'right': false,
  };
  String statusMessage = 'Start by taking photos of your vehicle';
  bool isLoading = false;

  @override
  void initState() {
    super.initState();
    _createSession();
  }

  Future<void> _createSession() async {
    setState(() {
      isLoading = true;
      statusMessage = 'Creating session...';
    });

    try {
      final response = await http.post(Uri.parse('$apiBaseUrl/session'));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          sessionId = data['session_id'];
          statusMessage = 'Session created. Please upload vehicle images.';
        });
      } else {
        setState(() {
          statusMessage = 'Failed to create session. Please try again.';
        });
      }
    } catch (e) {
      setState(() {
        statusMessage = 'Error: $e';
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<void> _uploadImage(String side) async {
    if (sessionId == null) {
      setState(() {
        statusMessage = 'No active session. Please restart the app.';
      });
      return;
    }

    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(
      source: ImageSource.camera,
      maxWidth: 1200,
      maxHeight: 1200,
      imageQuality: 85,
    );

    if (image == null) {
      return; // User cancelled the picker
    }

    setState(() {
      isLoading = true;
      statusMessage = 'Uploading $side view...';
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiBaseUrl/upload/$sessionId'),
      );

      request.files.add(await http.MultipartFile.fromPath(
        'file',
        image.path,
      ));

      request.fields['claimed_side'] = side;

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);
      var result = jsonDecode(response.body);

      if (result['status'] == 'success') {
        setState(() {
          uploadedSides[side] = true;
          
          if (result['is_complete']) {
            statusMessage = 'All images uploaded successfully! Your vehicle has been verified.';
          } else {
            final missingList = (result['missing_sides'] as List).join(', ');
            statusMessage = 'Successfully uploaded $side view. Still need: $missingList';
          }
        });
      } else {
        setState(() {
          statusMessage = 'Error: ${result['message']}';
        });
      }
    } catch (e) {
      setState(() {
        statusMessage = 'Upload error: $e';
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<void> _checkStatus() async {
    if (sessionId == null) {
      setState(() {
        statusMessage = 'No active session. Please restart the app.';
      });
      return;
    }

    setState(() {
      isLoading = true;
      statusMessage = 'Checking status...';
    });

    try {
      final response = await http.get(Uri.parse('$apiBaseUrl/status/$sessionId'));
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        
        // Update the UI based on uploaded sides
        for (var side in data['uploaded_sides']) {
          uploadedSides[side] = true;
        }
        
        setState(() {
          if (data['is_complete']) {
            statusMessage = 'All images uploaded successfully!';
          } else {
            final missingList = (data['missing_sides'] as List).join(', ');
            statusMessage = 'Still need to upload: $missingList';
          }
        });
      } else {
        setState(() {
          statusMessage = 'Failed to check status. Please try again.';
        });
      }
    } catch (e) {
      setState(() {
        statusMessage = 'Error: $e';
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Vehicle Image Validation'),
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Text(
                        statusMessage,
                        style: const TextStyle(fontSize: 16),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),
                  Text(
                    'Upload Vehicle Images',
                    style: Theme.of(context).textTheme.headline6,
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 16),
                  GridView.count(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    crossAxisCount: 2,
                    mainAxisSpacing: 10,
                    crossAxisSpacing: 10,
                    children: [
                      _buildImageUploadCard('Front', 'front'),
                      _buildImageUploadCard('Rear', 'rear'),
                      _buildImageUploadCard('Left Side', 'left'),
                      _buildImageUploadCard('Right Side', 'right'),
                    ],
                  ),
                  const SizedBox(height: 24),
                  ElevatedButton(
                    onPressed: _checkStatus,
                    child: const Text('Check Status'),
                  ),
                ],
              ),
            ),
    );
  }

  Widget _buildImageUploadCard(String title, String side) {
    final bool isUploaded = uploadedSides[side] ?? false;
    
    return Card(
      color: isUploaded ? Colors.green[50] : null,
      child: InkWell(
        onTap: isUploaded ? null : () => _uploadImage(side),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              isUploaded ? Icons.check_circle : Icons.add_a_photo,
              size: 40,
              color: isUploaded ? Colors.green : Colors.blue,
            ),
            const SizedBox(height: 8),
            Text(title),
            Text(
              isUploaded ? 'Uploaded' : 'Tap to upload',
              style: TextStyle(
                fontSize: 12,
                color: isUploaded ? Colors.green : Colors.grey,
              ),
            ),
          ],
        ),
      ),
    );
  }
} 