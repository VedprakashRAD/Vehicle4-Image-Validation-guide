import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'vehicle_image_uploader.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() async {
  // Ensure Flutter is initialized
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize the camera plugin early
  final cameras = await availableCameras();
  
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  
  const MyApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vehicle Inspection App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        appBarTheme: AppBarTheme(
          elevation: 0,
          centerTitle: true,
          backgroundColor: Colors.white,
          foregroundColor: Colors.black87,
          iconTheme: IconThemeData(color: Colors.black87),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            padding: EdgeInsets.symmetric(vertical: 12),
          ),
        ),
        cardTheme: CardTheme(
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
      home: HomeScreen(cameras: cameras),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomeScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  
  const HomeScreen({Key? key, required this.cameras}) : super(key: key);

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final String apiBaseUrl = 'http://localhost:9000'; // Replace with your FastAPI URL
  bool _isLoading = false;
  String? _sessionId;
  String _statusMessage = 'Welcome to Vehicle Inspection App';

  @override
  void initState() {
    super.initState();
    // Check for existing session or create a new one
    _checkOrCreateSession();
  }

  Future<void> _checkOrCreateSession() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Initializing session...';
    });
    
    try {
      // Create a new session
      final response = await http.post(
        Uri.parse('$apiBaseUrl/session'),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _sessionId = data['session_id'];
          _statusMessage = 'Session created. Ready to start vehicle inspection.';
        });
      } else {
        setState(() {
          _statusMessage = 'Error creating session. Please try again.';
        });
      }
    } catch (e) {
      print('Error checking/creating session: $e');
      setState(() {
        _statusMessage = 'Network error. Please check your connection.';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _startInspection() async {
    if (_sessionId == null) {
      setState(() {
        _statusMessage = 'No active session. Creating new session...';
      });
      await _checkOrCreateSession();
      if (_sessionId == null) return; // Still no session
    }
    
    // Navigate to the image uploader screen
    final result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => VehicleImageUploader(
          sessionId: _sessionId!,
        ),
      ),
    );
    
    // If result is true, inspection was completed
    if (result == true) {
      setState(() {
        _statusMessage = 'Vehicle inspection completed successfully! Creating new session...';
      });
      // Create a new session for the next inspection
      await _checkOrCreateSession();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Vehicle Inspection'),
      ),
      body: _isLoading
          ? Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // App logo or vehicle illustration
                  Icon(
                    Icons.directions_car,
                    size: 100,
                    color: Colors.blue,
                  ),
                  
                  SizedBox(height: 32),
                  
                  // App title
                  Text(
                    'Vehicle Image Validation',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  
                  SizedBox(height: 16),
                  
                  // App description
                  Text(
                    'Capture all sides of your vehicle with guided camera assistance',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey[700],
                    ),
                    textAlign: TextAlign.center,
                  ),
                  
                  SizedBox(height: 32),
                  
                  // Status message
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Text(
                        _statusMessage,
                        style: TextStyle(fontSize: 16),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                  
                  SizedBox(height: 32),
                  
                  // Session ID
                  if (_sessionId != null)
                    Text(
                      'Session ID: $_sessionId',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  
                  SizedBox(height: 32),
                  
                  // Start inspection button
                  ElevatedButton.icon(
                    onPressed: _startInspection,
                    icon: Icon(Icons.camera_alt),
                    label: Text(
                      'START INSPECTION',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                  
                  SizedBox(height: 16),
                  
                  // New session button
                  OutlinedButton.icon(
                    onPressed: _checkOrCreateSession,
                    icon: Icon(Icons.refresh),
                    label: Text('NEW SESSION'),
                    style: OutlinedButton.styleFrom(
                      padding: EdgeInsets.symmetric(vertical: 16),
                      side: BorderSide(color: Colors.blue),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                ],
              ),
            ),
    );
  }
} 