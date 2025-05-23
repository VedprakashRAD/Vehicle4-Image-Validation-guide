#!/usr/bin/env python3
import http.server
import socketserver
import os
import socket

# Configuration
PORT = 9090
APK_FILE = "app-debug.apk"

def get_local_ip():
    """Get the local IP address"""
    try:
        # Create a socket to determine the IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Connect to Google's DNS
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

class APKHandler(http.server.BaseHTTPRequestHandler):
    def _set_headers(self):
        """Set common headers for both GET and HEAD requests"""
        if not os.path.exists(APK_FILE):
            self.send_error(404, f"File {APK_FILE} not found")
            return False

        # Get file size
        file_size = os.path.getsize(APK_FILE)
        
        # Send headers
        self.send_response(200)
        self.send_header('Content-Type', 'application/vnd.android.package-archive')
        self.send_header('Content-Disposition', f'attachment; filename="{APK_FILE}"')
        self.send_header('Content-Length', str(file_size))
        self.end_headers()
        return True

    def do_HEAD(self):
        """Handle HEAD requests by sending headers only"""
        if self.path == '/' or self.path == '/download':
            try:
                self._set_headers()
            except Exception as e:
                self.send_error(500, f"Internal server error: {str(e)}")
        else:
            self.send_error(404, "File not found")

    def do_GET(self):
        """Handle GET requests by serving the APK file"""
        if self.path == '/' or self.path == '/download':
            try:
                # Set headers
                if not self._set_headers():
                    return
                
                # Send file content
                with open(APK_FILE, 'rb') as file:
                    self.wfile.write(file.read())
                    
                print(f"APK file served successfully to {self.client_address[0]}")
                
            except Exception as e:
                self.send_error(500, f"Internal server error: {str(e)}")
        else:
            self.send_error(404, "File not found")

def main():
    """Main function"""
    # Check if APK file exists
    if not os.path.exists(APK_FILE):
        print(f"Error: {APK_FILE} not found!")
        return
    
    # Get IP address
    ip_address = get_local_ip()
    
    # Print download URL
    print("\n=== DOWNLOAD LINK ===")
    print(f"http://{ip_address}:{PORT}/download")
    
    # Start server
    try:
        server = socketserver.TCPServer(("0.0.0.0", PORT), APKHandler)
        print(f"\nServer started on port {PORT}")
        print("Press Ctrl+C to stop the server")
        server.serve_forever()
    except Exception as e:
        print(f"Failed to start server: {e}")

if __name__ == "__main__":
    main() 