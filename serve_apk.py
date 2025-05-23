#!/usr/bin/env python3
import http.server
import socketserver
import os
import threading
import socket

# Configuration
APK_FILE = "app-debug.apk"
PORTS = [8000, 8080, 9090]  # Try multiple ports

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

def start_server(port):
    """Start HTTP server on specified port"""
    try:
        handler = CustomHTTPRequestHandler
        httpd = socketserver.TCPServer(("0.0.0.0", port), handler)
        print(f"Server started on port {port}")
        httpd.serve_forever()
    except Exception as e:
        print(f"Failed to start server on port {port}: {e}")

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
        for port in PORTS:
            print(f"http://{ip}:{port}/download")
    
    # Start servers on different ports
    threads = []
    for port in PORTS:
        thread = threading.Thread(target=start_server, args=(port,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    print("\nPress Ctrl+C to stop the servers")
    
    # Wait for all threads
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nShutting down servers...")

if __name__ == "__main__":
    main() 