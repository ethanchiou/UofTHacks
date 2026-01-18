#!/usr/bin/env python3
"""
Standalone Camera Viewer with Face/Hand Detection
Run with: python3 camera_viewer.py

This is a lightweight version that works on macOS without Raspberry Pi dependencies.
"""

import cv2
import time
import threading
import argparse
from typing import List, Optional
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe available - face mesh and hand tracking enabled")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("✗ MediaPipe not available - install with: pip3 install mediapipe")


@dataclass
class FaceData:
    detected: bool
    position: tuple
    size: float
    head_pose: dict = None


@dataclass
class HandData:
    detected: bool
    handedness: str
    position: tuple
    gesture: str
    fingers_up: list
    is_pinching: bool = False


class VisionService:
    """Lightweight vision service for camera processing"""
    
    def __init__(self, camera_index=0, resolution=(640, 480), fps=30):
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self._running = False
        self._thread = None
        
        # Detection results
        self.faces: List[FaceData] = []
        self.hands: List[HandData] = []
        self._lock = threading.Lock()
        
        # MediaPipe setup
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None
            self.hands_detector = None
            # Fallback to Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def start(self):
        """Start the vision service"""
        if self._running:
            return
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"✓ Camera started (index={self.camera_index}, resolution={self.resolution})")
        return True
    
    def stop(self):
        """Stop the vision service"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print("Camera stopped")
    
    def _process_loop(self):
        """Main processing loop"""
        frame_delay = 1.0 / self.fps
        
        while self._running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            faces = []
            hands = []
            
            if MEDIAPIPE_AVAILABLE:
                # Process with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face detection
                face_results = self.face_mesh.process(rgb_frame)
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        face_data = self._process_face(face_landmarks, frame.shape)
                        faces.append(face_data)
                
                # Hand detection
                hand_results = self.hands_detector.process(rgb_frame)
                if hand_results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                        handedness = "Right"
                        if hand_results.multi_handedness and idx < len(hand_results.multi_handedness):
                            handedness = hand_results.multi_handedness[idx].classification[0].label
                        hand_data = self._process_hand(hand_landmarks, frame.shape, handedness)
                        hands.append(hand_data)
            else:
                # Fallback to Haar cascade (faces only)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                for (x, y, w, h) in detected_faces:
                    frame_h, frame_w = frame.shape[:2]
                    center_x = x + w // 2
                    center_y = y + h // 2
                    pos_x = (center_x - frame_w // 2) / (frame_w // 2)
                    pos_y = (center_y - frame_h // 2) / (frame_h // 2)
                    size = (w * h) / (frame_w * frame_h)
                    faces.append(FaceData(True, (pos_x, pos_y), size, None))
            
            # Update results
            with self._lock:
                self.faces = faces
                self.hands = hands
            
            # FPS control
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
    
    def _process_face(self, landmarks, frame_shape) -> FaceData:
        """Process MediaPipe face landmarks"""
        frame_h, frame_w = frame_shape[:2]
        nose = landmarks.landmark[1]
        
        pos_x = (nose.x * frame_w - frame_w / 2) / (frame_w / 2)
        pos_y = (nose.y * frame_h - frame_h / 2) / (frame_h / 2)
        
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        eye_distance = np.sqrt(
            ((left_eye.x - right_eye.x) * frame_w) ** 2 +
            ((left_eye.y - right_eye.y) * frame_h) ** 2
        )
        size = min(1.0, eye_distance / 100.0)
        
        return FaceData(True, (pos_x, pos_y), size, None)
    
    def _process_hand(self, landmarks, frame_shape, handedness) -> HandData:
        """Process MediaPipe hand landmarks"""
        frame_h, frame_w = frame_shape[:2]
        wrist = landmarks.landmark[0]
        
        pos_x = (wrist.x * frame_w - frame_w / 2) / (frame_w / 2)
        pos_y = (wrist.y * frame_h - frame_h / 2) / (frame_h / 2)
        
        # Count fingers
        fingers_up = self._count_fingers(landmarks)
        
        # Pinch detection
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        dx = (thumb_tip.x - index_tip.x) * frame_w
        dy = (thumb_tip.y - index_tip.y) * frame_h
        distance = np.sqrt(dx**2 + dy**2)
        is_pinching = distance < 40
        
        # Gesture detection
        gesture = self._detect_gesture(fingers_up, is_pinching)
        
        return HandData(True, handedness, (pos_x, pos_y), gesture, fingers_up, is_pinching)
    
    def _count_fingers(self, landmarks) -> list:
        """Count raised fingers"""
        fingers = []
        
        # Thumb
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        pinky_mcp = landmarks.landmark[17]
        dist_tip = np.sqrt((thumb_tip.x - pinky_mcp.x)**2 + (thumb_tip.y - pinky_mcp.y)**2)
        dist_ip = np.sqrt((thumb_ip.x - pinky_mcp.x)**2 + (thumb_ip.y - pinky_mcp.y)**2)
        fingers.append(1 if dist_tip > dist_ip else 0)
        
        # Other fingers
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(1 if landmarks.landmark[tip].y < landmarks.landmark[pip].y else 0)
        
        return fingers
    
    def _detect_gesture(self, fingers_up: list, is_pinching: bool) -> str:
        """Detect hand gesture"""
        if is_pinching:
            return "Pinch"
        
        total = sum(fingers_up)
        if total == 0:
            return "Fist"
        elif total == 1 and fingers_up[1] == 1:
            return "Point"
        elif total == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:
            return "Peace"
        elif total == 5:
            return "Open"
        elif fingers_up[0] == 1 and total == 1:
            return "Thumbs Up"
        else:
            return f"{total} Fingers"
    
    def get_frame_with_overlay(self) -> Optional[bytes]:
        """Get current frame with detection overlay as JPEG"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        with self._lock:
            faces = self.faces.copy()
            hands = self.hands.copy()
        
        frame_h, frame_w = frame.shape[:2]
        
        # Draw faces
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for idx, face in enumerate(faces):
            if not face.detected:
                continue
            
            color = colors[idx % len(colors)]
            center_x = int((face.position[0] + 1.0) * frame_w / 2)
            center_y = int((face.position[1] + 1.0) * frame_h / 2)
            box_size = int(face.size * 200)
            
            if box_size > 0:
                x1 = max(0, center_x - box_size // 2)
                y1 = max(0, center_y - box_size // 2)
                x2 = min(frame_w, center_x + box_size // 2)
                y2 = min(frame_h, center_y + box_size // 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
                
                label = f"Face {idx + 1}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw hands
        hand_colors = [(0, 255, 255), (255, 165, 0)]
        for idx, hand in enumerate(hands):
            if not hand.detected:
                continue
            
            color = hand_colors[idx % len(hand_colors)]
            center_x = int((hand.position[0] + 1.0) * frame_w / 2)
            center_y = int((hand.position[1] + 1.0) * frame_h / 2)
            
            cv2.circle(frame, (center_x, center_y), 10, color, 2)
            
            label = f"{hand.handedness}: {hand.gesture}"
            cv2.putText(frame, label, (center_x - 50, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Summary
        info = f"Faces: {len(faces)} | Hands: {len(hands)}"
        cv2.putText(frame, info, (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()


# Global vision service
vision_service: Optional[VisionService] = None


class StreamHandler(BaseHTTPRequestHandler):
    """HTTP handler for video streaming"""
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def do_GET(self):
        if self.path == '/':
            self.send_html_page()
        elif self.path.startswith('/video_feed'):
            self.send_video_feed()
        elif self.path == '/status':
            self.send_status()
        else:
            self.send_error(404)
    
    def send_html_page(self):
        """Send the camera viewer HTML page"""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Camera Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
            max-width: 900px;
            width: 100%;
        }
        h1 { text-align: center; margin-bottom: 20px; color: #333; }
        .video-container {
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            aspect-ratio: 4/3;
            margin-bottom: 20px;
        }
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .status {
            display: flex;
            justify-content: space-around;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
        }
        .status-item { text-align: center; }
        .status-label { font-size: 12px; color: #666; }
        .status-value { font-size: 24px; font-weight: bold; color: #333; }
        .info { margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 8px; }
        .info h3 { margin-bottom: 10px; color: #2e7d32; }
        .info ul { margin-left: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📷 Camera Viewer</h1>
        <div class="video-container">
            <img src="/video_feed" alt="Camera feed">
        </div>
        <div class="status">
            <div class="status-item">
                <div class="status-label">Status</div>
                <div class="status-value" style="color: #4caf50;">Live</div>
            </div>
            <div class="status-item">
                <div class="status-label">Detection</div>
                <div class="status-value">''' + ('MediaPipe' if MEDIAPIPE_AVAILABLE else 'Haar') + '''</div>
            </div>
        </div>
        <div class="info">
            <h3>Detection Features:</h3>
            <ul>
                <li><strong>Faces:</strong> Up to 5 faces with colored boxes</li>
                <li><strong>Hands:</strong> Up to 2 hands with gesture recognition</li>
                <li><strong>Gestures:</strong> Fist, Point, Peace, Thumbs Up, Pinch, Open</li>
            </ul>
        </div>
    </div>
</body>
</html>'''
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_video_feed(self):
        """Send MJPEG video stream"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        
        try:
            while True:
                if vision_service:
                    frame_data = vision_service.get_frame_with_overlay()
                    if frame_data:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')
                time.sleep(0.033)  # ~30fps
        except (BrokenPipeError, ConnectionResetError):
            pass
    
    def send_status(self):
        """Send JSON status"""
        import json
        status = {
            'running': vision_service is not None and vision_service._running,
            'mediapipe': MEDIAPIPE_AVAILABLE,
            'faces': len(vision_service.faces) if vision_service else 0,
            'hands': len(vision_service.hands) if vision_service else 0
        }
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())


def main():
    global vision_service
    
    parser = argparse.ArgumentParser(description='Camera Viewer with Face/Hand Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--port', type=int, default=8080, help='HTTP port (default: 8080)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[640, 480], help='Resolution (default: 640 480)')
    parser.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Camera Viewer with Face/Hand Detection")
    print("=" * 60)
    
    # Start vision service
    vision_service = VisionService(
        camera_index=args.camera,
        resolution=tuple(args.resolution),
        fps=args.fps
    )
    
    if not vision_service.start():
        print("Failed to start camera. Exiting.")
        return
    
    # Start HTTP server
    server = HTTPServer(('0.0.0.0', args.port), StreamHandler)
    print(f"\n✓ Server running at http://localhost:{args.port}")
    print(f"✓ Open your browser to view the camera feed")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        vision_service.stop()
        server.shutdown()
        print("Done!")


if __name__ == '__main__':
    main()
