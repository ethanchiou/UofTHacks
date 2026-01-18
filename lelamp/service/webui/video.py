"""
Video streaming service for WebUI.

Provides MJPEG video feed with face, hand, and paper detection overlay.
"""

import time
import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, HTMLResponse

import lelamp.globals as g

router = APIRouter()


def draw_face_overlay(frame: np.ndarray, face_data, show_box: bool = True) -> np.ndarray:
    """Draw face detection overlay on frame."""
    if not face_data or not face_data.detected:
        return frame

    frame_h, frame_w = frame.shape[:2]

    # Convert normalized position to pixel coordinates
    pos_x, pos_y = face_data.position
    center_x = int((pos_x + 1.0) * frame_w / 2)
    center_y = int((pos_y + 1.0) * frame_h / 2)

    # Draw bounding box based on face size
    box_size = int(face_data.size * 200)
    if show_box and box_size > 0:
        x1 = max(0, center_x - box_size // 2)
        y1 = max(0, center_y - box_size // 2)
        x2 = min(frame_w, center_x + box_size // 2)
        y2 = min(frame_h, center_y + box_size // 2)

        # Draw green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw center crosshair
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)

    # Draw head pose if available
    if hasattr(face_data, "head_pose") and face_data.head_pose:
        pitch = face_data.head_pose["pitch"]
        yaw = face_data.head_pose["yaw"]
        roll = face_data.head_pose["roll"]
        text = f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def draw_hand_overlay(frame: np.ndarray, hand_data, show_landmarks: bool = True) -> np.ndarray:
    """Draw hand detection overlay on frame."""
    if not hand_data or not hand_data.detected:
        return frame

    frame_h, frame_w = frame.shape[:2]

    # Draw hand landmarks if available
    if show_landmarks and hand_data.landmarks:
        # Define hand connections for skeleton
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

        # Convert normalized landmarks to pixel coordinates
        points = []
        for lm in hand_data.landmarks:
            px = int(lm[0] * frame_w)
            py = int(lm[1] * frame_h)
            points.append((px, py))

        # Draw connections (skeleton)
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], (255, 165, 0), 2)

        # Draw landmarks as circles
        for i, point in enumerate(points):
            # Fingertips are indices 4, 8, 12, 16, 20
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(frame, point, 6, (0, 255, 255), -1)  # Yellow for fingertips
            else:
                cv2.circle(frame, point, 4, (255, 165, 0), -1)  # Orange for other joints

    # Draw wrist position indicator
    pos_x, pos_y = hand_data.position
    wrist_x = int((pos_x + 1.0) * frame_w / 2)
    wrist_y = int((pos_y + 1.0) * frame_h / 2)

    # Highlight if pinching
    if hand_data.is_pinching:
        cv2.circle(frame, (wrist_x, wrist_y), 15, (0, 0, 255), 3)

    return frame


def draw_paper_overlay(frame: np.ndarray, paper_data, show_contour: bool = True) -> np.ndarray:
    """Draw paper/document detection overlay on frame."""
    if not paper_data or not paper_data.detected:
        return frame

    frame_h, frame_w = frame.shape[:2]

    # Draw paper contour/corners
    if show_contour and paper_data.corners and len(paper_data.corners) >= 4:
        # Draw filled polygon with transparency
        pts = np.array(paper_data.corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 255))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw contour outline
        cv2.polylines(frame, [pts], True, (0, 200, 255), 3)
        
        # Draw corner points
        for i, corner in enumerate(paper_data.corners):
            cv2.circle(frame, corner, 8, (255, 100, 0), -1)
            cv2.circle(frame, corner, 8, (255, 255, 255), 2)

    # Draw center crosshair (tracking target)
    pos_x, pos_y = paper_data.position
    center_x = int((pos_x + 1.0) * frame_w / 2)
    center_y = int((pos_y + 1.0) * frame_h / 2)
    
    # Large crosshair for tracking visualization
    crosshair_size = 30
    cv2.line(frame, (center_x - crosshair_size, center_y), 
             (center_x + crosshair_size, center_y), (0, 255, 255), 2)
    cv2.line(frame, (center_x, center_y - crosshair_size), 
             (center_x, center_y + crosshair_size), (0, 255, 255), 2)
    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 255), 2)
    cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)

    # Draw frame center reference (where we want the paper to be)
    frame_center_x, frame_center_y = frame_w // 2, frame_h // 2
    cv2.drawMarker(frame, (frame_center_x, frame_center_y), (255, 0, 255), 
                   cv2.MARKER_CROSS, 20, 1)

    # Draw info text
    angle_text = f"Angle: {paper_data.angle:.1f}deg"
    size_text = f"Size: {paper_data.size*100:.1f}%"
    cv2.putText(frame, angle_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.putText(frame, size_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    return frame


def draw_status_overlay(frame: np.ndarray, face_data, hand_data, paper_data=None) -> np.ndarray:
    """Draw status information overlay on frame."""
    frame_h, frame_w = frame.shape[:2]

    # Determine status bar height based on what we're showing
    status_height = 110 if paper_data is not None else 80

    # Status background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame_h - status_height), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Paper status (top of status bar when enabled)
    if paper_data is not None:
        if paper_data.detected:
            paper_status = f"DETECTED (Area: {paper_data.size*100:.1f}%)"
            paper_color = (0, 200, 255)  # Cyan/orange for paper
        else:
            paper_status = "Not detected"
            paper_color = (128, 128, 128)
        cv2.putText(frame, f"Paper: {paper_status}", (10, frame_h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, paper_color, 2)

    # Face status
    face_status = "DETECTED" if (face_data and face_data.detected) else "Not detected"
    face_color = (0, 255, 0) if (face_data and face_data.detected) else (128, 128, 128)
    cv2.putText(frame, f"Face: {face_status}", (10, frame_h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

    # Hand status
    if hand_data and hand_data.detected:
        hand_status = f"DETECTED ({hand_data.handedness})"
        if hand_data.is_pinching:
            hand_status += " - PINCHING"
        hand_color = (0, 255, 255)
    else:
        hand_status = "Not detected"
        hand_color = (128, 128, 128)
    cv2.putText(frame, f"Hand: {hand_status}", (10, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

    # Timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, timestamp, (frame_w - 100, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def generate_video_feed(
    show_face: bool = True, 
    show_hand: bool = True, 
    show_paper: bool = True,
    show_status: bool = True
):
    """Generate video frames with face, hand, and paper detection overlay.
    Uses g.vision_service directly to always get the current service.
    
    Args:
        show_face: Show face detection box and crosshair
        show_hand: Show hand skeleton and landmarks
        show_paper: Show paper detection contour and center
        show_status: Show status bar at bottom
    """
    no_camera_frame = None

    while True:
        # Get vision service from globals (updated dynamically)
        vs = g.vision_service

        if not vs or not vs.cap:
            # Return a blank frame if no camera, but keep the stream open
            if no_camera_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "No Camera Connected", (150, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(blank, "Waiting for vision service...", (140, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)
                _, buffer = cv2.imencode(".jpg", blank)
                no_camera_frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + no_camera_frame + b"\r\n")
            time.sleep(0.5)  # Check less frequently when no camera
            continue

        ret, frame = vs.cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Get detection data
        face_data = vs.get_face_data()
        hand_data = vs.get_hand_data()
        paper_data = vs.get_paper_data() if hasattr(vs, 'get_paper_data') else None

        # Draw overlays (paper first so it's behind other overlays)
        if show_paper and paper_data:
            frame = draw_paper_overlay(frame, paper_data, show_contour=True)
        
        if show_face:
            frame = draw_face_overlay(frame, face_data, show_box=True)
        
        if show_hand:
            frame = draw_hand_overlay(frame, hand_data, show_landmarks=True)
        
        if show_status:
            frame = draw_status_overlay(frame, face_data, hand_data, 
                                        paper_data if show_paper else None)

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        time.sleep(0.033)  # ~30fps


@router.get("/video_feed")
async def video_feed(
    show_face: bool = False,  # Default off - focus on paper
    show_hand: bool = False,  # Default off - focus on paper
    show_paper: bool = True,  # Default on - paper tracking mode
    show_status: bool = True,
    show_box: bool = True  # Backward compatibility
):
    """Stream video with face, hand, and paper detection overlay (MJPEG).
    
    Query params:
        show_face: Show face detection overlay (default: false)
        show_hand: Show hand detection overlay (default: false)
        show_paper: Show paper detection overlay (default: true)
        show_status: Show status bar (default: true)
    """
    return StreamingResponse(
        generate_video_feed(
            show_face=show_face and show_box, 
            show_hand=show_hand,
            show_paper=show_paper,
            show_status=show_status
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# Monitoring page HTML
MONITOR_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeLamp Paper Tracking Monitor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 1.8rem;
            font-weight: 600;
            background: linear-gradient(90deg, #00c8ff, #ff8c00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(0, 200, 255, 0.1);
            border: 1px solid rgba(0, 200, 255, 0.3);
            border-radius: 20px;
            font-size: 0.9rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: #00c8ff;
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .video-container {
            background: #0d1117;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .video-header {
            padding: 16px 20px;
            background: rgba(255,255,255,0.02);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .video-title {
            font-size: 1rem;
            font-weight: 500;
            color: #e5e5e5;
        }
        
        .controls {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .control-btn {
            padding: 6px 12px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            color: #fff;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .control-btn:hover {
            background: rgba(255,255,255,0.2);
        }
        
        .control-btn.active {
            background: rgba(0, 200, 255, 0.2);
            border-color: #00c8ff;
            color: #00c8ff;
        }
        
        .control-btn.active.paper {
            background: rgba(255, 140, 0, 0.2);
            border-color: #ff8c00;
            color: #ff8c00;
        }
        
        .video-wrapper {
            position: relative;
            background: #000;
        }
        
        #videoFeed {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .info-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .info-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
        }
        
        .info-card h3 {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #888;
            margin-bottom: 8px;
        }
        
        .info-card .value {
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .info-card .value.green { color: #00ff88; }
        .info-card .value.cyan { color: #00c8ff; }
        .info-card .value.orange { color: #ff8c00; }
        .info-card .value.yellow { color: #ffd000; }
        .info-card .value.gray { color: #666; }
        
        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
            text-align: center;
            color: #666;
            font-size: 0.85rem;
        }
        
        .help-text {
            margin-top: 20px;
            padding: 16px;
            background: rgba(255, 140, 0, 0.05);
            border: 1px solid rgba(255, 140, 0, 0.2);
            border-radius: 8px;
            font-size: 0.9rem;
            color: #aaa;
        }
        
        .help-text code {
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }
        
        .tracking-info {
            margin-top: 20px;
            padding: 16px;
            background: rgba(0, 200, 255, 0.05);
            border: 1px solid rgba(0, 200, 255, 0.2);
            border-radius: 8px;
        }
        
        .tracking-info h4 {
            color: #00c8ff;
            margin-bottom: 10px;
        }
        
        .tracking-info ul {
            list-style: none;
            color: #aaa;
            font-size: 0.9rem;
        }
        
        .tracking-info li {
            margin-bottom: 5px;
            padding-left: 20px;
            position: relative;
        }
        
        .tracking-info li:before {
            content: ">";
            position: absolute;
            left: 0;
            color: #00c8ff;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Paper Tracking Monitor</h1>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>Live Feed</span>
            </div>
        </header>
        
        <div class="video-container">
            <div class="video-header">
                <span class="video-title">Camera Feed - Paper Detection & Tracking</span>
                <div class="controls">
                    <button class="control-btn active paper" onclick="toggleOverlay('paper', this)">Paper</button>
                    <button class="control-btn" onclick="toggleOverlay('face', this)">Face</button>
                    <button class="control-btn" onclick="toggleOverlay('hand', this)">Hand</button>
                    <button class="control-btn active" onclick="toggleOverlay('status', this)">Status</button>
                    <button class="control-btn" onclick="toggleFullscreen()">Fullscreen</button>
                </div>
            </div>
            <div class="video-wrapper">
                <img id="videoFeed" src="/video_feed?show_paper=true&show_face=false&show_hand=false&show_status=true" alt="Video Feed">
            </div>
        </div>
        
        <div class="info-panel">
            <div class="info-card">
                <h3>Mode</h3>
                <div class="value orange">Paper Track</div>
            </div>
            <div class="info-card">
                <h3>Connection</h3>
                <div class="value green" id="connectionStatus">Connected</div>
            </div>
            <div class="info-card">
                <h3>Stream URL</h3>
                <div class="value" style="font-size: 0.8rem; word-break: break-all;" id="streamUrl"></div>
            </div>
            <div class="info-card">
                <h3>Format</h3>
                <div class="value cyan">MJPEG</div>
            </div>
        </div>
        
        <div class="tracking-info">
            <h4>Paper Tracking Guide</h4>
            <ul>
                <li>Place a sheet of paper on your desk within camera view</li>
                <li>Orange outline shows detected paper boundaries</li>
                <li>Yellow crosshair marks the paper center (tracking target)</li>
                <li>Pink cross marks the frame center (where camera aims)</li>
                <li>Camera will move to keep paper centered in frame</li>
            </ul>
        </div>
        
        <div class="help-text">
            <strong>Tips:</strong> For best detection, use white/light paper on a darker surface. 
            Toggle overlays using buttons above. Access raw feed at <code>/video_feed</code> with params:
            <code>?show_paper=true&show_face=false&show_hand=false</code>
        </div>
        
        <footer>
            LeLamp Vision Service - Paper Tracking Mode
        </footer>
    </div>
    
    <script>
        // Set stream URL display
        document.getElementById('streamUrl').textContent = window.location.origin + '/video_feed';
        
        // Overlay state - paper tracking mode by default
        let overlays = { paper: true, face: false, hand: false, status: true };
        
        function updateVideoSrc() {
            const params = new URLSearchParams({
                show_paper: overlays.paper,
                show_face: overlays.face,
                show_hand: overlays.hand,
                show_status: overlays.status
            });
            document.getElementById('videoFeed').src = '/video_feed?' + params.toString();
        }
        
        function toggleOverlay(type, btn) {
            overlays[type] = !overlays[type];
            btn.classList.toggle('active');
            updateVideoSrc();
        }
        
        function toggleFullscreen() {
            const video = document.getElementById('videoFeed');
            if (video.requestFullscreen) {
                video.requestFullscreen();
            } else if (video.webkitRequestFullscreen) {
                video.webkitRequestFullscreen();
            }
        }
        
        // Monitor connection
        const img = document.getElementById('videoFeed');
        img.onerror = () => {
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').className = 'value gray';
            // Retry after 2 seconds
            setTimeout(() => {
                img.src = img.src.split('?')[0] + '?' + new URLSearchParams({
                    show_paper: overlays.paper,
                    show_face: overlays.face,
                    show_hand: overlays.hand,
                    show_status: overlays.status,
                    _t: Date.now()
                }).toString();
            }, 2000);
        };
        
        img.onload = () => {
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'value green';
        };
    </script>
</body>
</html>
"""


@router.get("/monitor")
async def monitor_page():
    """Serve the vision monitoring webpage."""
    return HTMLResponse(content=MONITOR_HTML)
