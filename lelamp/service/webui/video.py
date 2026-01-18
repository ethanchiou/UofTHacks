"""
Video streaming service for WebUI.

Provides MJPEG video feed with face detection overlay.
"""

import time
import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

import lelamp.globals as g

router = APIRouter()


def draw_face_overlay(frame: np.ndarray, face_data, show_box: bool = True) -> np.ndarray:
    """Draw face detection overlay on frame (single face, backward compatibility)."""
    if face_data and face_data.detected:
        return draw_faces_overlay(frame, [face_data], show_box)
    return frame

def draw_faces_overlay(frame: np.ndarray, faces_data: list, show_box: bool = True) -> np.ndarray:
    """Draw multiple face detection overlays on frame."""
    if not faces_data:
        return frame

    frame_h, frame_w = frame.shape[:2]
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]

    for idx, face_data in enumerate(faces_data):
        if not face_data or not face_data.detected:
            continue

        color = colors[idx % len(colors)]
        
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

            # Draw colored box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw center crosshair
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), color, 2)
            cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), color, 2)

            # Draw face label
            label = f"Face {idx + 1}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw head pose if available
        if hasattr(face_data, "head_pose") and face_data.head_pose:
            pitch = face_data.head_pose["pitch"]
            yaw = face_data.head_pose["yaw"]
            roll = face_data.head_pose["roll"]
            text = f"F{idx+1}: Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"
            cv2.putText(frame, text, (10, 30 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def draw_hands_overlay(frame: np.ndarray, hands_data: list, show_box: bool = True) -> np.ndarray:
    """Draw multiple hand detection overlays on frame."""
    if not hands_data:
        return frame

    frame_h, frame_w = frame.shape[:2]
    colors = [
        (0, 255, 255),  # Yellow
        (255, 165, 0),  # Orange
    ]

    for idx, hand_data in enumerate(hands_data):
        if not hand_data or not hand_data.detected:
            continue

        color = colors[idx % len(colors)]
        
        # Convert normalized position to pixel coordinates
        pos_x, pos_y = hand_data.position
        center_x = int((pos_x + 1.0) * frame_w / 2)
        center_y = int((pos_y + 1.0) * frame_h / 2)

        # Draw hand center
        cv2.circle(frame, (center_x, center_y), 10, color, 2)
        cv2.circle(frame, (center_x, center_y), 3, color, -1)

        # Draw hand label with gesture
        label = f"{hand_data.handedness} Hand: {hand_data.gesture}"
        if hand_data.is_pinching:
            label += " (Pinching)"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = max(0, center_x - label_size[0] // 2)
        text_y = center_y - 20
        
        # Draw label background
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - label_size[1] - 5),
            (text_x + label_size[0] + 5, text_y + 5),
            color,
            -1
        )
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw fingers count
        fingers_text = f"Fingers: {sum(hand_data.fingers_up)}"
        cv2.putText(
            frame,
            fingers_text,
            (text_x, text_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

    return frame


def generate_video_feed(show_box: bool = True):
    """Generate video frames with face detection overlay.
    Uses g.vision_service directly to always get the current service.
    """
    no_camera_frame = None

    while True:
        # Get vision service from globals (updated dynamically)
        vs = g.vision_service

        if not vs or not vs.cap:
            # Return a blank frame if no camera, but keep the stream open
            if no_camera_frame is None:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, "No Camera", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode(".jpg", blank)
                no_camera_frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + no_camera_frame + b"\r\n")
            time.sleep(0.5)  # Check less frequently when no camera
            continue

        ret, frame = vs.cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Get multiple faces and hands data
        faces_data = vs.get_faces_data()
        hands_data = vs.get_hands_data()
        
        if show_box:
            # Draw all faces
            frame = draw_faces_overlay(frame, faces_data, show_box=True)
            # Draw all hands
            frame = draw_hands_overlay(frame, hands_data, show_box=True)
            
            # Draw summary info
            info_text = f"Faces: {len(faces_data)} | Hands: {len(hands_data)}"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        time.sleep(0.033)  # ~30fps


@router.get("/video_feed")
async def video_feed(show_box: bool = True):
    """Stream video with face detection overlay (MJPEG)."""
    return StreamingResponse(
        generate_video_feed(show_box=show_box),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
