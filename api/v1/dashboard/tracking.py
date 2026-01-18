"""
Face tracking and vision endpoints.

Provides face detection status and tracking control.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any

from api.deps import get_vision_service, load_config, save_config

router = APIRouter()

# Module-level state for tracking
_tracking_enabled = False
_latest_stats: Dict[str, Any] = {
    'fps': 0,
    'face_detected': False,
    'position': (0.0, 0.0),
    'size': 0.0,
    'head_pose': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},
    'timestamp': 0.0
}


def get_tracking_stats() -> Dict[str, Any]:
    """Get current tracking stats (used by websocket)."""
    return _latest_stats.copy()


def is_tracking_enabled() -> bool:
    """Check if tracking is enabled."""
    return _tracking_enabled


@router.get("/status")
async def tracking_status():
    """Get tracking status."""
    vision = get_vision_service()

    if not vision:
        return {'enabled': False, 'available': False}

    return {
        'enabled': _tracking_enabled,
        'available': True
    }


@router.post("/enable")
async def enable_tracking():
    """Enable face tracking mode."""
    global _tracking_enabled, _latest_stats

    vision = get_vision_service()
    if not vision:
        return {'success': False, 'error': 'Vision service not available'}

    if _tracking_enabled:
        return {'success': True, 'message': 'Already enabled'}

    # Callback that updates stats
    def track_callback(face_data):
        global _latest_stats
        _latest_stats.update({
            'face_detected': face_data.detected,
            'position': face_data.position,
            'size': face_data.size,
            'timestamp': face_data.timestamp
        })
        if hasattr(face_data, 'head_pose'):
            _latest_stats['head_pose'] = face_data.head_pose

    vision.enable_tracking_mode(track_callback)
    _tracking_enabled = True

    return {'success': True, 'message': 'Face tracking enabled'}


@router.post("/disable")
async def disable_tracking():
    """Disable face tracking mode."""
    global _tracking_enabled

    vision = get_vision_service()
    if not vision:
        return {'success': False, 'error': 'Vision service not available'}

    vision.disable_tracking_mode()
    _tracking_enabled = False

    return {'success': True, 'message': 'Face tracking disabled'}


@router.get("/config")
async def get_tracking_config():
    """Get face tracking configuration."""
    try:
        config = load_config()
        return {
            "success": True,
            "face_tracking": config.get('face_tracking', {}),
            "vision": config.get('vision', {})
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/config")
async def update_tracking_config(data: Dict[str, Any]):
    """Update face tracking configuration."""
    try:
        config = load_config()

        if 'face_tracking' in data:
            config.setdefault('face_tracking', {})
            config['face_tracking'].update(data['face_tracking'])

        if 'vision' in data:
            config.setdefault('vision', {})
            config['vision'].update(data['vision'])

        save_config(config)
        return {"success": True, "message": "Tracking config updated"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/vision")
async def get_vision_data():
    """Get current vision data including multiple faces and hands."""
    vision = get_vision_service()
    
    if not vision:
        return {
            'success': False,
            'error': 'Vision service not available',
            'faces_count': 0,
            'hands_count': 0,
            'faces': [],
            'hands': []
        }
    
    try:
        faces_data = vision.get_faces_data()
        hands_data = vision.get_hands_data()
        
        # Convert faces to dict format
        faces = []
        for face in faces_data:
            if face and face.detected:
                faces.append({
                    'position': list(face.position),
                    'size': face.size,
                    'head_pose': face.head_pose if face.head_pose else None,
                    'timestamp': face.timestamp
                })
        
        # Convert hands to dict format
        hands = []
        for hand in hands_data:
            if hand and hand.detected:
                hands.append({
                    'handedness': hand.handedness,
                    'position': list(hand.position),
                    'gesture': hand.gesture,
                    'is_pinching': hand.is_pinching,
                    'fingers_up': hand.fingers_up,
                    'timestamp': hand.timestamp
                })
        
        return {
            'success': True,
            'faces_count': len(faces),
            'hands_count': len(hands),
            'faces': faces,
            'hands': hands
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'faces_count': 0,
            'hands_count': 0,
            'faces': [],
            'hands': []
        }
