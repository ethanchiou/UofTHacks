"""
LeLamp API - Clean, modular REST API

Structure:
    api/
    ├── __init__.py          # This file - app factory
    ├── deps.py              # Dependency injection
    └── v1/
        ├── setup/           # Setup wizard endpoints
        ├── dashboard/       # Dashboard/monitoring endpoints
        ├── agent/           # Agent control (wake/sleep)
        ├── spotify/         # Spotify OAuth and playback
        └── ...

Usage:
    from api import create_api
    app = create_api()
"""

import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv

import lelamp.globals as g
from lelamp.user_data import get_env_path


def create_api(
    title: str = "LeLamp API",
    version: str = "1.0.0",
    vision_service=None,
) -> FastAPI:
    """
    Factory function to create the FastAPI application.

    Args:
        title: API title
        version: API version
        vision_service: Optional vision service instance for video streaming

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title=title,
        version=version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Share vision service with globals
    if vision_service is not None:
        g.vision_service = vision_service

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    from api.v1 import router as v1_router
    app.include_router(v1_router, prefix="/api/v1")

    # Include WebSocket handlers
    from api.websocket import router as ws_router
    app.include_router(ws_router, prefix="/ws", tags=["websocket"])

    # Include video streaming
    from lelamp.service.webui.video import router as video_router
    app.include_router(video_router, tags=["video"])

    # Camera viewer page
    @app.get("/camera", response_class=HTMLResponse)
    async def camera_viewer():
        """Simple camera viewer web page."""
        return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeLamp Camera Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 30px;
            max-width: 1200px;
            width: 100%;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        
        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            aspect-ratio: 4 / 3;
            margin-bottom: 20px;
        }
        
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        
        .status {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .status-item {
            text-align: center;
        }
        
        .status-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .status-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        
        .status-value.active {
            color: #10b981;
        }
        
        .status-value.inactive {
            color: #ef4444;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-secondary {
            background: #e5e7eb;
            color: #333;
        }
        
        .btn-danger {
            background: #ef4444;
            color: white;
        }
        
        .error {
            background: #fee2e2;
            color: #991b1b;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📷 LeLamp Camera Viewer</h1>
        <p class="subtitle">Live camera feed with face detection</p>
        
        <div class="error" id="error"></div>
        
        <div class="video-container">
            <div class="loading" id="loading">
                <span class="spinner"></span>
                Connecting to camera...
            </div>
            <img id="videoStream" src="/video_feed?show_box=true" alt="Camera feed" style="display: none;">
        </div>
        
            <div class="status">
            <div class="status-item">
                <div class="status-label">Stream Status</div>
                <div class="status-value" id="streamStatus">Connecting...</div>
            </div>
            <div class="status-item">
                <div class="status-label">Faces Detected</div>
                <div class="status-value" id="facesStatus">0</div>
            </div>
            <div class="status-item">
                <div class="status-label">Hands Detected</div>
                <div class="status-value" id="handsStatus">0</div>
            </div>
            <div class="status-item">
                <div class="status-label">FPS</div>
                <div class="status-value" id="fpsStatus">—</div>
            </div>
        </div>
        
        <div id="detectionInfo" style="display: none; margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px;">
            <h3 style="margin-bottom: 10px; font-size: 16px;">Detection Details</h3>
            <div id="facesInfo"></div>
            <div id="handsInfo" style="margin-top: 10px;"></div>
        </div>
        
        <div class="controls">
            <button class="btn-primary" onclick="toggleFaceBox()">Toggle Face Box</button>
            <button class="btn-secondary" onclick="refreshStream()">Refresh Stream</button>
            <button class="btn-secondary" onclick="takeSnapshot()">Take Snapshot</button>
        </div>
    </div>
    
    <script>
        let showBox = true;
        let frameCount = 0;
        let lastFpsTime = Date.now();
        let fpsInterval;
        
        const videoStream = document.getElementById('videoStream');
        const loading = document.getElementById('loading');
        const errorDiv = document.getElementById('error');
        const streamStatus = document.getElementById('streamStatus');
        const faceStatus = document.getElementById('faceStatus');
        const fpsStatus = document.getElementById('fpsStatus');
        
        // Update stream status
        videoStream.onload = function() {
            loading.style.display = 'none';
            videoStream.style.display = 'block';
            streamStatus.textContent = 'Connected';
            streamStatus.className = 'status-value active';
            hideError();
        };
        
        videoStream.onerror = function() {
            loading.style.display = 'none';
            showError('Failed to load camera stream. Make sure the camera is connected and vision service is enabled.');
            streamStatus.textContent = 'Disconnected';
            streamStatus.className = 'status-value inactive';
        };
        
        // FPS counter
        function startFpsCounter() {
            fpsInterval = setInterval(() => {
                const now = Date.now();
                const elapsed = (now - lastFpsTime) / 1000;
                const fps = Math.round(frameCount / elapsed);
                fpsStatus.textContent = fps + ' fps';
                frameCount = 0;
                lastFpsTime = now;
            }, 1000);
        }
        
        videoStream.onload = function() {
            frameCount++;
            if (!fpsInterval) {
                startFpsCounter();
            }
            loading.style.display = 'none';
            videoStream.style.display = 'block';
            streamStatus.textContent = 'Connected';
            streamStatus.className = 'status-value active';
            hideError();
        };
        
        function toggleFaceBox() {
            showBox = !showBox;
            refreshStream();
        }
        
        function refreshStream() {
            loading.style.display = 'block';
            videoStream.style.display = 'none';
            videoStream.src = '/video_feed?show_box=' + showBox + '&t=' + Date.now();
        }
        
        function takeSnapshot() {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = videoStream.naturalWidth || 640;
            canvas.height = videoStream.naturalHeight || 480;
            ctx.drawImage(videoStream, 0, 0);
            
            canvas.toBlob(function(blob) {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'lelamp-snapshot-' + new Date().toISOString().replace(/:/g, '-') + '.jpg';
                a.click();
                URL.revokeObjectURL(url);
            }, 'image/jpeg', 0.95);
        }
        
        function showError(message) {
            errorDiv.textContent = message;
            errorDiv.classList.add('show');
        }
        
        function hideError() {
            errorDiv.classList.remove('show');
        }
        
        // Parse detection info from video stream (if available via API)
        async function updateDetectionInfo() {
            try {
                const response = await fetch('/api/v1/dashboard/tracking/vision');
                if (response.ok) {
                    const data = await response.json();
                    const facesCount = data.faces_count || 0;
                    const handsCount = data.hands_count || 0;
                    
                    document.getElementById('facesStatus').textContent = facesCount;
                    document.getElementById('facesStatus').className = facesCount > 0 ? 'status-value active' : 'status-value inactive';
                    
                    document.getElementById('handsStatus').textContent = handsCount;
                    document.getElementById('handsStatus').className = handsCount > 0 ? 'status-value active' : 'status-value inactive';
                    
                    if (facesCount > 0 || handsCount > 0) {
                        document.getElementById('detectionInfo').style.display = 'block';
                        
                        // Update faces info
                        if (data.faces && data.faces.length > 0) {
                            const facesHtml = data.faces.map((face, idx) => 
                                `Face ${idx + 1}: Position (${face.position[0].toFixed(2)}, ${face.position[1].toFixed(2)}), Size: ${(face.size * 100).toFixed(0)}%`
                            ).join('<br>');
                            document.getElementById('facesInfo').innerHTML = '<strong>Faces:</strong><br>' + facesHtml;
                        } else {
                            document.getElementById('facesInfo').innerHTML = '';
                        }
                        
                        // Update hands info
                        if (data.hands && data.hands.length > 0) {
                            const handsHtml = data.hands.map((hand, idx) => 
                                `${hand.handedness} Hand ${idx + 1}: ${hand.gesture}${hand.is_pinching ? ' (Pinching)' : ''}`
                            ).join('<br>');
                            document.getElementById('handsInfo').innerHTML = '<strong>Hands:</strong><br>' + handsHtml;
                        } else {
                            document.getElementById('handsInfo').innerHTML = '';
                        }
                    } else {
                        document.getElementById('detectionInfo').style.display = 'none';
                    }
                }
            } catch (e) {
                // API might not be available, that's okay
            }
        }
        
        // Update detection info every second
        setInterval(updateDetectionInfo, 1000);
        updateDetectionInfo();
        
        // Auto-refresh on connection issues
        setInterval(() => {
            if (videoStream.style.display === 'none' && loading.style.display === 'none') {
                refreshStream();
            }
        }, 5000);
    </script>
</body>
</html>
        """)

    # Health check endpoint
    @app.get("/api/health")
    async def health_check():
        return {"status": "ok", "version": version}

    # Paths
    base_dir = Path(__file__).parent.parent
    dist_dir = base_dir / "frontend" / "dist"
    assets_dir = base_dir / "assets"

    # Mount assets
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    # Mount dist-assets for React bundles
    dist_assets = dist_dir / "dist-assets"
    if dist_assets.exists():
        app.mount("/dist-assets", StaticFiles(directory=dist_assets), name="dist-assets")

    # Serve static files from dist root (favicon, lelamp.svg, etc.)
    @app.get("/lelamp.svg")
    async def serve_lelamp_svg():
        svg_path = dist_dir / "lelamp.svg"
        if svg_path.exists():
            return FileResponse(svg_path, media_type="image/svg+xml")
        return FileResponse(assets_dir / "images" / "lelamp.svg", media_type="image/svg+xml")

    @app.get("/vite.svg")
    async def serve_vite_svg():
        return FileResponse(dist_dir / "vite.svg", media_type="image/svg+xml")

    # Root route - redirect to dashboard or setup
    @app.get("/")
    async def root():
        """Redirect to dashboard or setup based on config."""
        env_path = get_env_path()
        if env_path.exists():
            load_dotenv(env_path, override=True)
        else:
            load_dotenv(override=True)

        cfg = g.CONFIG or {}
        setup = cfg.get("setup", {})

        if not setup.get("setup_complete", False):
            return RedirectResponse(url="/setup", status_code=302)

        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_key or len(openai_key) < 20:
            return RedirectResponse(url="/setup", status_code=302)

        if g.calibration_required:
            return RedirectResponse(url="/setup", status_code=302)

        return RedirectResponse(url="/dashboard", status_code=302)

    # Serve React SPA for frontend routes
    @app.get("/setup")
    @app.get("/dashboard")
    @app.get("/settings")
    async def serve_spa():
        """Serve the React SPA."""
        if dist_dir.exists() and (dist_dir / "index.html").exists():
            return FileResponse(dist_dir / "index.html")
        return HTMLResponse(
            content="<h1>LeLamp</h1><p>Frontend not built. Run: cd frontend && npm run build</p>"
        )

    # API docs redirect
    @app.get("/api/v1/docs")
    async def api_docs_redirect():
        """Redirect to main API documentation."""
        return RedirectResponse(url="/api/docs", status_code=302)

    return app


# Convenience export
__all__ = ["create_api"]
