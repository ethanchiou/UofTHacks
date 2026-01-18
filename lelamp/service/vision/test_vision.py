#!/usr/bin/env python3
"""
Test script for VisionService
Run with: uv run -m lelamp.service.vision.test_vision
Or with camera device: uv run -m lelamp.service.vision.test_vision --camera /dev/usbcam
"""

import time
import sys
import argparse
import logging
from lelamp.service.vision.vision_service import VisionService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Test the vision service"""
    parser = argparse.ArgumentParser(description='Test VisionService')
    parser.add_argument(
        '--camera',
        type=str,
        default='0',
        help='Camera device index or path (e.g., 0, 1, /dev/usbcam, /dev/video0)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        nargs=2,
        default=[640, 480],
        metavar=('WIDTH', 'HEIGHT'),
        help='Camera resolution (default: 640 480)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Target FPS (default: 20)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show all detection attempts, not just successful detections'
    )
    
    args = parser.parse_args()
    
    # Convert camera argument to int if it's a number, otherwise keep as string
    try:
        camera_index = int(args.camera)
    except ValueError:
        camera_index = args.camera
    
    print("=" * 60)
    print("VisionService Test")
    print("=" * 60)
    print(f"Camera: {camera_index}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print(f"FPS: {args.fps}")
    print("=" * 60)
    print("\nInitializing VisionService...")
    
    # Create vision service
    vision = VisionService(
        camera_index=camera_index,
        resolution=tuple(args.resolution),
        fps=args.fps
    )
    
    print("Starting vision service...")
    vision.start()
    
    # Give camera time to initialize
    print("Waiting for camera to initialize...")
    time.sleep(2)
    
    try:
        print("\n" + "=" * 60)
        print("Vision service running. Press Ctrl+C to stop.")
        print("=" * 60 + "\n")
        
        last_face_detected = False
        last_hand_detected = False
        
        iteration = 0
        while True:
            iteration += 1
            face_data = vision.get_face_data()
            hand_data = vision.get_hand_data()
            
            # Face detection
            if face_data:
                if face_data.detected:
                    if not last_face_detected or args.verbose:
                        print(f"[{iteration:4d}] ✓ Face detected!")
                        print(f"         Position: ({face_data.position[0]:+.2f}, {face_data.position[1]:+.2f})")
                        print(f"         Size: {face_data.size:.2f}")
                        if face_data.head_pose:
                            print(f"         Head pose - Pitch: {face_data.head_pose['pitch']:+.1f}°, "
                                  f"Yaw: {face_data.head_pose['yaw']:+.1f}°, "
                                  f"Roll: {face_data.head_pose['roll']:+.1f}°")
                    last_face_detected = True
                else:
                    if last_face_detected or args.verbose:
                        if args.verbose:
                            print(f"[{iteration:4d}] ✗ No face detected")
                    last_face_detected = False
            
            # Hand detection
            if hand_data:
                if hand_data.detected:
                    if not last_hand_detected or args.verbose:
                        print(f"[{iteration:4d}] ✓ Hand detected!")
                        print(f"         Gesture: {hand_data.gesture}")
                        print(f"         Handedness: {hand_data.handedness}")
                        print(f"         Pinching: {hand_data.is_pinching}")
                        print(f"         Position: ({hand_data.position[0]:+.2f}, {hand_data.position[1]:+.2f})")
                    last_hand_detected = True
                else:
                    if last_hand_detected and args.verbose:
                        print(f"[{iteration:4d}] ✗ No hand detected")
                    last_hand_detected = False
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping vision service...")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        vision.stop()
        print("Vision service stopped.")
        print("=" * 60)

if __name__ == "__main__":
    main()
