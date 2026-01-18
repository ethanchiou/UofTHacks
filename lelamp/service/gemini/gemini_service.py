"""
Gemini AI Service for LeLamp.

Provides joke detection and text analysis using Google's Gemini API.
"""

import json
import logging
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Run: pip install google-generativeai")


class GeminiService:
    """
    Gemini AI service for joke detection and analysis.
    
    Features:
    - Detect if user speech contains a joke
    - Rate humor level (1-10)
    - Classify joke type (pun, sarcasm, wordplay, etc.)
    """

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini service.
        
        Args:
            api_key: Google AI API key (or uses GEMINI_API_KEY env var)
            model: Gemini model to use (default: gemini-2.0-flash)
        """
        import os
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        # Get API key from param or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set GEMINI_API_KEY in .env")
        
        self.model_name = model
        self._model = None
        self._last_check_time = 0
        self._cooldown_seconds = 5  # Don't check too frequently
        
        # Initialize
        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(model)
        
        logger.info(f"GeminiService initialized with model: {model}")

    async def detect_joke(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to detect if it contains a joke.
        
        Args:
            text: User's speech text
            
        Returns:
            Dict with keys:
                - is_joke: bool
                - humor_level: int (1-10)
                - joke_type: str (pun/sarcasm/wordplay/observational/none)
        """
        import time
        
        # Cooldown check
        now = time.time()
        if now - self._last_check_time < self._cooldown_seconds:
            return {"is_joke": False, "humor_level": 0, "joke_type": "none", "skipped": True}
        
        self._last_check_time = now
        
        # Skip very short texts
        if len(text.split()) < 3:
            return {"is_joke": False, "humor_level": 0, "joke_type": "none"}
        
        prompt = f"""Analyze if this text contains a joke or humor. Return ONLY valid JSON, no other text:
{{"is_joke": true or false, "humor_level": 1-10, "joke_type": "pun" or "sarcasm" or "wordplay" or "observational" or "none"}}

Text: "{text}"

JSON:"""

        try:
            # Run in executor to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(prompt)
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Clean up response (remove markdown if present)
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            logger.debug(f"Joke detection result: {result}")
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            return {"is_joke": False, "humor_level": 0, "joke_type": "none", "error": str(e)}
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"is_joke": False, "humor_level": 0, "joke_type": "none", "error": str(e)}

    def set_cooldown(self, seconds: float):
        """Set cooldown between joke checks."""
        self._cooldown_seconds = max(0, seconds)
        logger.info(f"Joke detection cooldown set to {seconds}s")
