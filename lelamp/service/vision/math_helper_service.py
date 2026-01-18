"""
MathHelperService - Captures frames and solves math problems using Gemini Vision API.

When the user says "help" (or variations), this service:
1. Captures the current camera frame
2. Sends it to Google Gemini Vision API
3. Extracts and solves the math problem
4. Speaks the answer using TTS

Usage:
    from lelamp.service.vision.math_helper_service import MathHelperService
    
    math_helper = MathHelperService(api_key="your-gemini-api-key")
    math_helper.start()
    
    # When "help" is detected in transcript:
    answer = await math_helper.process_math_help()
    # Returns integer answer (default 4 if processing fails)
"""

import cv2
import base64
import logging
import threading
import asyncio
import time
import os
import sys
import traceback
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np

# Configure logging to show all messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MathHelperService")
logger.setLevel(logging.DEBUG)

# Try to import Google Generative AI
GENAI_AVAILABLE = False
genai = None
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    logger.info("google-generativeai imported successfully")
except ImportError as e:
    logger.error(f"google-generativeai not installed: {e}")
    logger.error("Run: pip install google-generativeai")

# Try to import PIL
PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
    logger.info("PIL imported successfully")
except ImportError as e:
    logger.error(f"PIL not installed: {e}")
    logger.error("Run: pip install Pillow")


@dataclass
class MathResult:
    """Result from math problem solving"""
    answer: int
    problem_text: str
    explanation: str
    confidence: float
    timestamp: float
    raw_response: str = ""


class MathHelperService:
    """
    Service that loads test.jpg and solves math problems using Gemini Vision API.
    
    Listens for "help" keyword variations in transcripts and automatically processes 
    the math problem from test.jpg file.
    """
    
    # Default Gemini API key
    DEFAULT_API_KEY = "AIzaSyBMwyjT3LjEPrWgdwXzmu3P8GSssEhF5Bo"
    
    # Path to test image (relative to UofTHacks directory)
    TEST_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "test.jpg")
    
    # Extended trigger phrases for homework help activation
    DEFAULT_TRIGGER_PHRASES = [
        # Direct help requests
        "help",
        "help me",
        "help with this",
        "help on this",
        "help with this problem",
        "help on this problem",
        "help me with this",
        "help me with this problem",
        "help solve",
        "help solve this",
        # Need help variations
        "i need help",
        "need help",
        "need help with this",
        "i need help with this",
        "i need help with this problem",
        "need some help",
        "i need some help",
        # Can you help variations
        "can you help",
        "can you help me",
        "can you help with this",
        "could you help",
        "would you help",
        "please help",
        "help please",
        # Homework specific
        "homework help",
        "help with homework",
        "help me with homework",
        "help with my homework",
        "i need help with homework",
        "help on homework",
        # Solve variations
        "solve this",
        "solve this problem",
        "solve the problem",
        "what's the answer",
        "what is the answer",
        "whats the answer",
        "figure this out",
        "work this out",
        # Math specific
        "solve the math",
        "math problem",
        "math help",
        "calculate this",
        "what does this equal",
        "do the math",
        "help with math",
        # Question words
        "what is this",
        "how do i solve",
        "how to solve",
        # Camera activation
        "look at this",
        "check this",
        "see this problem",
        "read this",
        "scan this",
    ]
    
    def __init__(
        self,
        api_key: str = None,
        trigger_phrases: list = None,
        default_answer: int = 4,
        cooldown_seconds: float = 2.0
    ):
        """
        Initialize the Math Helper Service.
        
        Args:
            api_key: Google Gemini API key (uses default if not provided)
            trigger_phrases: List of phrases that trigger math help
            default_answer: Default answer if processing fails (default: 4)
            cooldown_seconds: Minimum time between triggers (default: 2.0)
        """
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.trigger_phrases = trigger_phrases or self.DEFAULT_TRIGGER_PHRASES
        self.default_answer = default_answer
        self.cooldown_seconds = cooldown_seconds
        
        self._running = False
        self._last_trigger_time = 0.0
        self._processing = False
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_trigger_callback: Optional[Callable] = None
        self._on_result_callback: Optional[Callable[[MathResult], None]] = None
        self._speak_callback: Optional[Callable[[str], None]] = None
        
        # Latest result for debugging
        self.latest_result: Optional[MathResult] = None
        self.last_error: Optional[str] = None
        self.last_frame: Optional[np.ndarray] = None
        
        # Initialize Gemini
        self.model = None
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize the Gemini API client."""
        if not GENAI_AVAILABLE:
            logger.error("=" * 50)
            logger.error("GEMINI API NOT AVAILABLE")
            logger.error("Install with: pip install google-generativeai")
            logger.error("=" * 50)
            return
        
        try:
            logger.info(f"Configuring Gemini with API key: {self.api_key[:10]}...{self.api_key[-4:]}")
            genai.configure(api_key=self.api_key)
            
            # Use gemini-2.0-flash-lite for better rate limits on free tier
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
            logger.info("Gemini model initialized: gemini-2.0-flash-lite")
            logger.info("Skipping API test to conserve quota - will test on first actual request")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            logger.error(traceback.format_exc())
            self.model = None
    
    def start(self):
        """Start the math helper service."""
        self._running = True
        logger.info("=" * 50)
        logger.info("MathHelperService STARTED")
        logger.info(f"Trigger phrases: {len(self.trigger_phrases)} configured")
        logger.info(f"Gemini available: {self.model is not None}")
        logger.info("=" * 50)
    
    def stop(self):
        """Stop the math helper service."""
        self._running = False
        logger.info("MathHelperService stopped")
    
    def set_speak_callback(self, callback: Callable[[str], None]):
        """Set callback function for speaking the answer."""
        self._speak_callback = callback
        logger.info("Speak callback registered")
    
    def set_result_callback(self, callback: Callable[[MathResult], None]):
        """Set callback for when a math result is ready."""
        self._on_result_callback = callback
    
    def check_transcript(self, transcript: str) -> bool:
        """
        Check if transcript contains trigger phrases.
        
        Args:
            transcript: The transcribed text to check
            
        Returns:
            True if trigger phrase found and not in cooldown
        """
        if not self._running:
            return False
        
        transcript_lower = transcript.lower().strip()
        
        # Quick check for "help" first (most common trigger)
        if "help" in transcript_lower:
            current_time = time.time()
            if current_time - self._last_trigger_time < self.cooldown_seconds:
                return False
            return True
        
        # Check other trigger phrases
        for phrase in self.trigger_phrases:
            if phrase.lower() in transcript_lower:
                current_time = time.time()
                if current_time - self._last_trigger_time < self.cooldown_seconds:
                    return False
                return True
        
        return False
    
    def load_test_image(self) -> Optional[np.ndarray]:
        """
        Load the test.jpg image file.
        
        Returns:
            numpy array of the image, or None if loading failed
        """
        # Try multiple paths to find test.jpg
        possible_paths = [
            self.TEST_IMAGE_PATH,
            "/Users/ethanchiou/Desktop/Programming/Projects/LeLamp/UofTHacks/test.jpg",
            os.path.join(os.getcwd(), "test.jpg"),
            "test.jpg",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                frame = cv2.imread(path)
                if frame is not None:
                    self.last_frame = frame.copy()
                    return frame
        
        logger.error("Could not find test.jpg")
        return None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Load the test.jpg image (legacy method name for compatibility).
        
        Returns:
            numpy array of the image, or None if loading failed
        """
        return self.load_test_image()
    
    def frame_to_base64(self, frame: np.ndarray, quality: int = 85) -> str:
        """Convert a frame to base64 encoded JPEG."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64 = base64.b64encode(buffer).decode('utf-8')
        logger.debug(f"Frame encoded to base64: {len(b64)} characters")
        return b64
    
    async def analyze_math_problem(self, frame: np.ndarray) -> MathResult:
        """
        Send frame to Gemini API and extract math problem answer.
        
        Args:
            frame: The camera frame containing the math problem
            
        Returns:
            MathResult with the answer
        """
        if not GENAI_AVAILABLE or not PIL_AVAILABLE or self.model is None:
            self.last_error = "Gemini not available"
            return MathResult(
                answer=self.default_answer, problem_text="", explanation="Gemini not available",
                confidence=0.0, timestamp=time.time(), raw_response=""
            )
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Create the prompt
            prompt = """Look at this image. Find and solve the math problem shown.
IMPORTANT: Return ONLY the numerical answer as a single integer.
If you see "2 + 2 = ?", respond with just: 4
Look at the image and give me ONLY the integer answer:"""

            # Call Gemini API
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, image]
            )
            
            if response is None:
                self.last_error = "Gemini returned None"
                return MathResult(
                    answer=self.default_answer, problem_text="", explanation="No response",
                    confidence=0.0, timestamp=time.time(), raw_response=""
                )
            
            # Get response text
            try:
                response_text = response.text.strip()
            except Exception:
                response_text = str(response)
            
            # Extract integer from response
            answer = self._extract_integer(response_text)
            
            result = MathResult(
                answer=answer,
                problem_text=response_text,
                explanation=response_text,
                confidence=0.9 if answer != self.default_answer else 0.5,
                timestamp=time.time(),
                raw_response=response_text
            )
            
            self.latest_result = result
            self.last_error = None
            return result
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            self.last_error = str(e)
            return MathResult(
                answer=self.default_answer, problem_text="", explanation=str(e),
                confidence=0.0, timestamp=time.time(), raw_response=""
            )
    
    def _extract_integer(self, text: str) -> int:
        """Extract an integer from text response."""
        import re
        
        # Clean the text
        text = text.strip()
        
        # If the entire response is just a number
        if re.match(r'^-?\d+$', text):
            return int(text)
        
        # Try to find numbers in the text
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^-?\d+$', line):
                return int(line)
        
        # Look for common patterns
        patterns = [
            r'(?:answer|result|equals?|=|is)\s*[:=]?\s*(-?\d+)',
            r'(-?\d+)\s*$',
            r'^(-?\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Find all numbers and return the last one
        numbers = re.findall(r'-?\d+', text)
        if numbers:
            return int(numbers[-1])
        
        logger.warning(f"Could not extract integer, using default: {self.default_answer}")
        return self.default_answer
    
    async def process_math_help(self) -> tuple[int, str]:
        """
        Main method to process a math help request.
        
        Loads test.jpg, analyzes with Gemini, outputs answer as:
        1. Text transcript (printed and logged)
        2. Audio through speakers
        
        Returns:
            Tuple of (answer: int, transcript: str)
        """
        with self._lock:
            if self._processing:
                return self.default_answer, f"The answer is {self.default_answer}"
            self._processing = True
        
        answer = self.default_answer
        transcript = ""
        
        try:
            self._last_trigger_time = time.time()
            
            # Load test.jpg
            frame = self.load_test_image()
            
            if frame is None:
                logger.error("Could not load test.jpg")
                transcript = f"The answer is {self.default_answer}"
            else:
                # Analyze with Gemini
                result = await self.analyze_math_problem(frame)
                self.latest_result = result
                answer = result.answer
                transcript = f"The answer is {answer}"
                
                # Callback if set
                if self._on_result_callback:
                    try:
                        self._on_result_callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
            # Set light GREEN for answer output
            await self._set_light_green()
            
            # Audio output
            if self._speak_callback:
                try:
                    self._speak_callback(transcript)
                except Exception:
                    await self._speak_with_audio_service(transcript)
            else:
                await self._speak_with_audio_service(transcript)
            
            # Keep light green for 5 seconds total, then restore
            await asyncio.sleep(5)
            await self._restore_light()
            
            return answer, transcript
            
        except Exception as e:
            logger.error(f"Error in process_math_help: {e}")
            logger.error(traceback.format_exc())
            transcript = f"An error occurred. The default answer is {self.default_answer}"
            return self.default_answer, transcript
            
        finally:
            with self._lock:
                self._processing = False
    
    async def _set_light_green(self):
        """Set the RGB light to green color."""
        try:
            import lelamp.globals as g
            if g.rgb_service:
                # Store current color to restore later
                self._previous_color = g.rgb_service.controller.get_current_color()
                # Set to bright green
                g.rgb_service.controller.set_color((0, 255, 0), transition=False)
                logger.debug("Light set to GREEN")
        except Exception as e:
            logger.debug(f"Could not set light green: {e}")
    
    async def _restore_light(self):
        """Restore the RGB light to previous color."""
        try:
            import lelamp.globals as g
            if g.rgb_service:
                # Restore previous color or default to off
                color = getattr(self, '_previous_color', (0, 0, 0))
                g.rgb_service.controller.set_color(color, transition=True)
                logger.debug("Light restored")
        except Exception as e:
            logger.debug(f"Could not restore light: {e}")
    
    async def _speak_with_audio_service(self, text: str):
        """
        Speak text using available TTS methods.
        Prioritizes Raspberry Pi audio service for speaker output.
        """
        logger.info(f"Attempting to speak on Raspberry Pi: '{text}'")
        
        # Try to import globals for audio service
        try:
            import lelamp.globals as g
            audio_svc = g.audio_service
        except ImportError:
            audio_svc = None
        
        # Method 1: gTTS + audio_service (Raspberry Pi)
        try:
            from gtts import gTTS
            import tempfile
            
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_path = f.name
                tts.save(temp_path)
            
            if audio_svc:
                audio_svc.play_by_path(temp_path, blocking=True)
                os.unlink(temp_path)
                return
            else:
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    data, sr = sf.read(temp_path)
                    sd.play(data, sr)
                    sd.wait()
                    os.unlink(temp_path)
                    return
                except Exception:
                    os.unlink(temp_path)
        except Exception:
            pass
        
        # Method 2: espeak (Linux/Raspberry Pi)
        try:
            import subprocess
            result = subprocess.run(['espeak', text], capture_output=True, timeout=10)
            if result.returncode == 0:
                return
        except Exception:
            pass
        
        # Method 3: pico2wave (Linux)
        try:
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            result = subprocess.run(['pico2wave', '-w', temp_path, text], capture_output=True, timeout=10)
            if result.returncode == 0:
                if audio_svc:
                    audio_svc.play_by_path(temp_path, blocking=True)
                else:
                    subprocess.run(['aplay', temp_path], timeout=10)
                os.unlink(temp_path)
                return
        except Exception:
            pass
        
        # Method 4: macOS say (fallback for dev)
        try:
            import subprocess
            result = subprocess.run(['say', text], capture_output=True, timeout=10)
            if result.returncode == 0:
                return
        except Exception:
            pass


# Global instance
_math_helper_instance: Optional[MathHelperService] = None


def get_math_helper() -> MathHelperService:
    """Get or create the global MathHelperService instance."""
    global _math_helper_instance
    if _math_helper_instance is None:
        _math_helper_instance = MathHelperService()
    return _math_helper_instance


def init_math_helper(api_key: str = None) -> MathHelperService:
    """Initialize the global MathHelperService instance."""
    global _math_helper_instance
    _math_helper_instance = MathHelperService(api_key=api_key)
    _math_helper_instance.start()
    return _math_helper_instance


# =============================================================================
# COMPREHENSIVE TEST FUNCTION
# =============================================================================

async def run_full_test():
    """
    Run a comprehensive test of the MathHelperService.
    Tests each component individually with test.jpg.
    """
    print("\n" + "=" * 70)
    print("   MATH HELPER SERVICE - COMPREHENSIVE TEST")
    print("   Using test.jpg for image analysis")
    print("=" * 70)
    
    # Test 1: Check dependencies
    print("\n[TEST 1] Checking dependencies...")
    print(f"  - google-generativeai: {'✓ INSTALLED' if GENAI_AVAILABLE else '✗ NOT INSTALLED'}")
    print(f"  - PIL/Pillow: {'✓ INSTALLED' if PIL_AVAILABLE else '✗ NOT INSTALLED'}")
    
    if not GENAI_AVAILABLE:
        print("\n  ❌ ERROR: Install google-generativeai with: pip install google-generativeai")
        return
    
    if not PIL_AVAILABLE:
        print("\n  ❌ ERROR: Install Pillow with: pip install Pillow")
        return
    
    print("  ✓ PASS: All dependencies installed")
    
    # Test 2: Initialize service
    print("\n[TEST 2] Initializing MathHelperService...")
    service = MathHelperService()
    service.start()
    
    if service.model is None:
        print("  ❌ ERROR: Gemini model failed to initialize")
        print(f"  Last error: {service.last_error}")
        return
    
    print("  ✓ PASS: Service initialized successfully")
    print(f"  Test image path: {service.TEST_IMAGE_PATH}")
    
    # Test 3: Test Gemini API with text-only request
    print("\n[TEST 3] Testing Gemini API (text-only)...")
    try:
        response = service.model.generate_content("What is 2 + 2? Reply with just the number.")
        print(f"  Response: '{response.text.strip()}'")
        print("  ✓ PASS: Gemini API responding")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return
    
    # Test 4: Test trigger phrase detection
    print("\n[TEST 4] Testing trigger phrase detection...")
    test_phrases = [
        ("help me with this problem", True),
        ("i need help", True),
        ("can you help", True),
        ("solve this", True),
        ("homework help", True),
        ("help with my homework", True),
        ("what's the weather", False),
        ("hello world", False),
    ]
    
    all_passed = True
    for phrase, expected in test_phrases:
        service._last_trigger_time = 0  # Reset cooldown
        result = service.check_transcript(phrase)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} '{phrase}' -> {result} (expected {expected})")
    
    if all_passed:
        print("  ✓ PASS: All trigger phrases working correctly")
    
    # Test 5: Test loading test.jpg
    print("\n[TEST 5] Testing test.jpg loading...")
    frame = service.load_test_image()
    
    if frame is None:
        print("  ❌ ERROR: Could not load test.jpg")
        print("  Make sure test.jpg exists in the UofTHacks directory")
        return
    else:
        print(f"  ✓ PASS: test.jpg loaded - shape={frame.shape}")
    
    # Save copy for verification
    debug_path = "/tmp/math_helper_test_input.jpg"
    cv2.imwrite(debug_path, frame)
    print(f"  Saved copy to: {debug_path}")
    
    # Test 6: Test Gemini with test.jpg
    print("\n[TEST 6] Testing Gemini API with test.jpg...")
    print("  Sending image to Gemini...")
    result = await service.analyze_math_problem(frame)
    
    print(f"\n  📊 GEMINI RESPONSE:")
    print(f"  Raw response: '{result.raw_response}'")
    print(f"  Extracted answer: {result.answer}")
    print(f"  Confidence: {result.confidence}")
    
    if result.raw_response:
        print("  ✓ PASS: Gemini processed image and returned response")
    else:
        print("  ❌ WARNING: No response from Gemini")
    
    # Test 7: Full pipeline with TEXT + AUDIO output
    print("\n[TEST 7] Testing full pipeline (TEXT + AUDIO)...")
    print("  This will output the answer both as text and through speakers...")
    service._last_trigger_time = 0  # Reset cooldown
    answer, transcript = await service.process_math_help()
    
    print(f"\n  📢 FINAL OUTPUT:")
    print(f"  Answer: {answer}")
    print(f"  Transcript: {transcript}")
    
    print("\n" + "=" * 70)
    print("   TEST COMPLETE")
    print("=" * 70)
    print(f"\n📋 Summary:")
    print(f"  Latest result: {service.latest_result}")
    print(f"  Last error: {service.last_error}")
    print(f"\n✅ If you heard the answer spoken aloud, audio output is working!")


# =============================================================================
# MAIN - Run test when executed directly
# =============================================================================

if __name__ == "__main__":
    print("Running MathHelperService test...")
    asyncio.run(run_full_test())
