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
    DEFAULT_API_KEY = "AIzaSyBCTkIxNDvXV-V1CPjPai0kxXNTTOlmImE"
    
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
            logger.debug("Service not running, ignoring transcript")
            return False
        
        transcript_lower = transcript.lower().strip()
        logger.debug(f"Checking transcript: '{transcript_lower}'")
        
        # Check for trigger phrases (check longer phrases first)
        sorted_phrases = sorted(self.trigger_phrases, key=len, reverse=True)
        
        for phrase in sorted_phrases:
            if phrase.lower() in transcript_lower:
                current_time = time.time()
                time_since_last = current_time - self._last_trigger_time
                
                # Check cooldown
                if time_since_last < self.cooldown_seconds:
                    logger.info(f"Trigger '{phrase}' found but in cooldown ({time_since_last:.1f}s < {self.cooldown_seconds}s)")
                    return False
                
                logger.info("=" * 50)
                logger.info(f"TRIGGER DETECTED: '{phrase}'")
                logger.info(f"Full transcript: '{transcript}'")
                logger.info("=" * 50)
                return True
        
        return False
    
    def load_test_image(self) -> Optional[np.ndarray]:
        """
        Load the test.jpg image file.
        
        Returns:
            numpy array of the image, or None if loading failed
        """
        logger.info("=" * 50)
        logger.info("LOADING TEST IMAGE")
        logger.info("=" * 50)
        
        # Try multiple paths to find test.jpg
        possible_paths = [
            self.TEST_IMAGE_PATH,
            "/Users/ethanchiou/Desktop/Programming/Projects/LeLamp/UofTHacks/test.jpg",
            os.path.join(os.getcwd(), "test.jpg"),
            "test.jpg",
        ]
        
        for path in possible_paths:
            logger.info(f"Trying path: {path}")
            if os.path.exists(path):
                logger.info(f"Found test.jpg at: {path}")
                frame = cv2.imread(path)
                if frame is not None:
                    self.last_frame = frame.copy()
                    logger.info(f"Image loaded successfully: shape={frame.shape}, dtype={frame.dtype}")
                    return frame
                else:
                    logger.error(f"cv2.imread failed for: {path}")
            else:
                logger.debug(f"Path does not exist: {path}")
        
        logger.error("Could not find test.jpg!")
        logger.error(f"Searched paths: {possible_paths}")
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
        logger.info("=" * 50)
        logger.info("ANALYZING MATH PROBLEM WITH GEMINI")
        logger.info("=" * 50)
        
        if not GENAI_AVAILABLE:
            error_msg = "Gemini API not available - google-generativeai not installed"
            logger.error(error_msg)
            self.last_error = error_msg
            return MathResult(
                answer=self.default_answer,
                problem_text="",
                explanation=error_msg,
                confidence=0.0,
                timestamp=time.time(),
                raw_response=""
            )
        
        if not PIL_AVAILABLE:
            error_msg = "PIL not available - Pillow not installed"
            logger.error(error_msg)
            self.last_error = error_msg
            return MathResult(
                answer=self.default_answer,
                problem_text="",
                explanation=error_msg,
                confidence=0.0,
                timestamp=time.time(),
                raw_response=""
            )
        
        if self.model is None:
            error_msg = "Gemini model not initialized"
            logger.error(error_msg)
            self.last_error = error_msg
            return MathResult(
                answer=self.default_answer,
                problem_text="",
                explanation=error_msg,
                confidence=0.0,
                timestamp=time.time(),
                raw_response=""
            )
        
        try:
            # Convert BGR to RGB
            logger.info(f"Converting frame: shape={frame.shape}, dtype={frame.dtype}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            logger.info(f"PIL Image created: size={image.size}, mode={image.mode}")
            
            # Save debug image
            debug_path = "/tmp/math_helper_debug.jpg"
            try:
                image.save(debug_path)
                logger.info(f"Debug image saved to: {debug_path}")
            except Exception as e:
                logger.warning(f"Could not save debug image: {e}")
            
            # Create the prompt
            prompt = """Look at this image. Find and solve the math problem shown.

IMPORTANT: Return ONLY the numerical answer as a single integer.

If you see "2 + 2 = ?", respond with just: 4
If you see "5 x 3 = ?", respond with just: 15
If you see "10 - 7 = ?", respond with just: 3
If you see "20 / 4 = ?", respond with just: 5

Look at the image and give me ONLY the integer answer:"""

            logger.info("Sending request to Gemini API...")
            logger.info(f"Prompt: {prompt[:100]}...")
            
            start_time = time.time()
            
            # Call Gemini API (synchronously in thread)
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, image]
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Gemini API response received in {elapsed:.2f}s")
            
            # Check response
            if response is None:
                error_msg = "Gemini returned None response"
                logger.error(error_msg)
                self.last_error = error_msg
                return MathResult(
                    answer=self.default_answer,
                    problem_text="",
                    explanation=error_msg,
                    confidence=0.0,
                    timestamp=time.time(),
                    raw_response=""
                )
            
            # Get response text
            try:
                response_text = response.text.strip()
            except Exception as e:
                error_msg = f"Could not get response text: {e}"
                logger.error(error_msg)
                # Try to get any available text
                response_text = str(response)
            
            logger.info("=" * 50)
            logger.info(f"GEMINI RAW RESPONSE: '{response_text}'")
            logger.info("=" * 50)
            
            # Extract integer from response
            answer = self._extract_integer(response_text)
            logger.info(f"Extracted answer: {answer}")
            
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
            error_msg = f"Error analyzing math problem: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.last_error = error_msg
            
            return MathResult(
                answer=self.default_answer,
                problem_text="",
                explanation=error_msg,
                confidence=0.0,
                timestamp=time.time(),
                raw_response=""
            )
    
    def _extract_integer(self, text: str) -> int:
        """Extract an integer from text response."""
        import re
        
        logger.debug(f"Extracting integer from: '{text}'")
        
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
        logger.info("=" * 60)
        logger.info("PROCESSING MATH HELP REQUEST")
        logger.info("=" * 60)
        
        with self._lock:
            if self._processing:
                logger.warning("Already processing a request, skipping")
                return self.default_answer, f"The answer is {self.default_answer}"
            self._processing = True
        
        answer = self.default_answer
        transcript = ""
        
        try:
            self._last_trigger_time = time.time()
            
            # Step 1: Load test.jpg
            logger.info("[Step 1/4] Loading test.jpg...")
            frame = self.load_test_image()
            
            if frame is None:
                logger.error("FAILED: Could not load test.jpg")
                answer = self.default_answer
                transcript = f"Sorry, I couldn't load the image. The default answer is {self.default_answer}"
            else:
                logger.info(f"SUCCESS: Image loaded - {frame.shape}")
                
                # Step 2: Analyze with Gemini
                logger.info("[Step 2/4] Sending to Gemini API...")
                result = await self.analyze_math_problem(frame)
                self.latest_result = result
                answer = result.answer
                logger.info(f"SUCCESS: Got answer - {answer}")
                
                # Build transcript with full context
                if result.explanation and result.explanation != str(answer):
                    transcript = f"The answer is {answer}"
                else:
                    transcript = f"The answer is {answer}"
                
                # Step 3: Callback
                if self._on_result_callback:
                    logger.info("[Step 3/4] Calling result callback...")
                    try:
                        self._on_result_callback(result)
                    except Exception as e:
                        logger.error(f"Result callback error: {e}")
            
            # Step 4: OUTPUT - Both Text AND Audio
            logger.info("[Step 4/4] Outputting answer (TEXT + AUDIO)...")
            
            # ===== TEXT OUTPUT =====
            # Print to console (visible transcript)
            print("\n")
            print("=" * 60)
            print("  📢 MATH HELPER RESPONSE")
            print("=" * 60)
            print(f"  TRANSCRIPT: {transcript}")
            print(f"  ANSWER: {answer}")
            print("=" * 60)
            print("\n")
            
            # Log for transcript record
            logger.info(f"TEXT OUTPUT - Transcript: {transcript}")
            logger.info(f"TEXT OUTPUT - Answer: {answer}")
            
            # ===== AUDIO OUTPUT =====
            logger.info("Starting audio output...")
            if self._speak_callback:
                try:
                    self._speak_callback(transcript)
                    logger.info("Audio output via speak_callback")
                except Exception as e:
                    logger.error(f"Speak callback error: {e}")
                    # Fallback to direct audio
                    await self._speak_with_audio_service(transcript)
            else:
                await self._speak_with_audio_service(transcript)
            
            logger.info("=" * 60)
            logger.info("MATH HELP COMPLETE")
            logger.info(f"  Answer: {answer}")
            logger.info(f"  Transcript: {transcript}")
            logger.info("=" * 60)
            
            return answer, transcript
            
        except Exception as e:
            logger.error(f"Error in process_math_help: {e}")
            logger.error(traceback.format_exc())
            transcript = f"An error occurred. The default answer is {self.default_answer}"
            return self.default_answer, transcript
            
        finally:
            with self._lock:
                self._processing = False
    
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
        
        # Method 1: Use gTTS + audio_service (Raspberry Pi speakers)
        # This is the PRIMARY method for Raspberry Pi output
        try:
            from gtts import gTTS
            import tempfile
            
            logger.info("Generating TTS audio with gTTS...")
            tts = gTTS(text=text, lang='en')
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_path = f.name
                tts.save(temp_path)
            
            logger.info(f"TTS audio saved to: {temp_path}")
            
            # Play via audio_service (Raspberry Pi speakers)
            if audio_svc:
                logger.info("Playing via audio_service (Raspberry Pi speaker)...")
                audio_svc.play_by_path(temp_path, blocking=True)
                logger.info("SUCCESS: Played on Raspberry Pi speaker via audio_service")
                os.unlink(temp_path)
                return
            else:
                logger.warning("audio_service not available, trying direct sounddevice...")
                # Fallback to sounddevice (plays on whatever device is default)
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    data, sr = sf.read(temp_path)
                    sd.play(data, sr)
                    sd.wait()
                    logger.info("Played with gTTS via sounddevice")
                    os.unlink(temp_path)
                    return
                except Exception as e:
                    logger.debug(f"sounddevice playback failed: {e}")
                    os.unlink(temp_path)
                    
        except ImportError:
            logger.debug("gTTS not installed, trying other methods...")
        except Exception as e:
            logger.warning(f"gTTS method failed: {e}")
        
        # Method 2: espeak (Linux/Raspberry Pi native TTS)
        # This speaks directly on the Pi without needing gTTS
        try:
            import subprocess
            logger.info("Trying espeak (Raspberry Pi native TTS)...")
            result = subprocess.run(['espeak', text], capture_output=True, timeout=10)
            if result.returncode == 0:
                logger.info("SUCCESS: Spoke with espeak on Raspberry Pi")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"espeak not available: {e}")
        
        # Method 3: pico2wave (Linux TTS alternative)
        try:
            import subprocess
            import tempfile
            
            logger.info("Trying pico2wave (Linux TTS)...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            # Generate speech with pico2wave
            result = subprocess.run(
                ['pico2wave', '-w', temp_path, text],
                capture_output=True, timeout=10
            )
            
            if result.returncode == 0:
                # Play via audio_service
                if audio_svc:
                    audio_svc.play_by_path(temp_path, blocking=True)
                    logger.info("SUCCESS: Played pico2wave audio on Raspberry Pi")
                else:
                    subprocess.run(['aplay', temp_path], timeout=10)
                    logger.info("Played pico2wave via aplay")
                os.unlink(temp_path)
                return
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"pico2wave not available: {e}")
        except Exception as e:
            logger.debug(f"pico2wave failed: {e}")
        
        # Method 4: macOS say command (ONLY for Mac development)
        # This is the FALLBACK when not on Raspberry Pi
        try:
            import subprocess
            logger.info("Falling back to macOS 'say' command (local Mac only)...")
            result = subprocess.run(['say', text], capture_output=True, timeout=10)
            if result.returncode == 0:
                logger.info("Spoke with macOS 'say' command (LOCAL Mac, not Raspberry Pi)")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"macOS say not available: {e}")
        
        logger.warning("No TTS method available - could not speak on any device")


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
