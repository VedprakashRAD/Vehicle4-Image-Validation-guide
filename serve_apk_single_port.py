#!/usr/bin/env python3
import http.server
import socketserver
import os
import socket

# Configuration
APK_FILE = "app-debug.apk"  # APK is in the current directory
PORT = 9090  # Only use port 9090

def get_ip_addresses():
    """Get all IP addresses of the machine"""
    ip_list = []
    try:
        # Get hostname
        hostname = socket.gethostname()
        
        # Get all IP addresses
        addresses = socket.getaddrinfo(hostname, None)
        
        for address in addresses:
            ip = address[4][0]
            # Filter out IPv6 and loopback addresses
            if ':' not in ip and ip != '127.0.0.1':
                ip_list.append(ip)
        
        # Add localhost as fallback
        if '127.0.0.1' not in ip_list:
            ip_list.append('127.0.0.1')
            
        # Remove duplicates
        ip_list = list(set(ip_list))
    except Exception as e:
        print(f"Error getting IP addresses: {e}")
        ip_list = ['127.0.0.1']  # Fallback to localhost
    
    return ip_list

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        # If path is root or /download, serve the APK
        if self.path == '/' or self.path == '/download':
            self.path = f'/{APK_FILE}'
            
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def main():
    """Main function"""
    # Check if APK file exists
    if not os.path.exists(APK_FILE):
        print(f"Error: {APK_FILE} not found!")
        return
    
    # Get IP addresses
    ip_addresses = get_ip_addresses()
    
    # Print download URLs
    print("\n=== DOWNLOAD LINKS ===")
    for ip in ip_addresses:
        print(f"http://{ip}:{PORT}/download")
    
    # Start server on port 9090 only
    try:
        handler = CustomHTTPRequestHandler
        httpd = socketserver.TCPServer(("0.0.0.0", PORT), handler)
        print(f"\nServer started on port {PORT}")
        print("Press Ctrl+C to stop the server")
        httpd.serve_forever()
    except Exception as e:
        print(f"Failed to start server on port {PORT}: {e}")

if __name__ == "__main__":
    main() 