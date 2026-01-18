"""
Joke Detection and Response Handler for LeLamp.

Orchestrates the joke detection → laugh response → memory storage pipeline.
"""

import logging
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class JokeHandler:
    """
    Handles joke detection and laugh responses.
    
    Flow:
    1. Receive transcribed text
    2. Send to Gemini for joke detection
    3. If joke detected, trigger laugh response (animation + sound + RGB)
    4. Store joke preference in Backboard memory
    """

    def __init__(
        self,
        gemini_service,
        backboard_service,
        animation_service,
        audio_service,
        rgb_service,
        min_humor_level: int = 5
    ):
        """
        Initialize joke handler.
        
        Args:
            gemini_service: GeminiService instance
            backboard_service: BackboardService instance
            animation_service: AnimationService instance
            audio_service: AudioService instance
            rgb_service: RGBService instance
            min_humor_level: Minimum humor level to trigger response (1-10)
        """
        self.gemini = gemini_service
        self.backboard = backboard_service
        self.animation = animation_service
        self.audio = audio_service
        self.rgb = rgb_service
        self.min_humor_level = min_humor_level
        self._enabled = True
        
        logger.info(f"JokeHandler initialized (min_humor_level={min_humor_level})")

    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process user text for joke detection and response.
        
        Args:
            text: Transcribed user speech
            
        Returns:
            Dict with detection result and actions taken
        """
        if not self._enabled:
            return {"processed": False, "reason": "disabled"}
        
        if not self.gemini:
            return {"processed": False, "reason": "no_gemini_service"}
        
        result = {
            "processed": True,
            "text": text,
            "is_joke": False,
            "actions": []
        }
        
        try:
            # Step 1: Detect joke with Gemini
            detection = await self.gemini.detect_joke(text)
            
            if detection.get("skipped"):
                result["reason"] = "cooldown"
                return result
            
            result["detection"] = detection
            result["is_joke"] = detection.get("is_joke", False)
            result["humor_level"] = detection.get("humor_level", 0)
            result["joke_type"] = detection.get("joke_type", "none")
            
            # Step 2: If joke detected and funny enough, respond
            if result["is_joke"] and result["humor_level"] >= self.min_humor_level:
                await self._trigger_laugh_response(result["humor_level"])
                result["actions"].append("laugh_response")
                
                # Step 3: Store in Backboard memory and get cumulative stats
                if self.backboard:
                    stats = await self.backboard.store_joke_reaction(
                        joke_text=text,
                        joke_type=result["joke_type"],
                        humor_level=result["humor_level"]
                    )
                    result["actions"].append("stored_memory")
                    result["cumulative_stats"] = stats
                    
                    logger.info(
                        f"Joke detected! type={result['joke_type']}, level={result['humor_level']} | "
                        f"Cumulative: {stats['total_jokes']} jokes, avg={stats['average_humor']}/10"
                    )
                else:
                    logger.info(f"Joke detected! type={result['joke_type']}, level={result['humor_level']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing joke: {e}")
            result["error"] = str(e)
            return result

    async def _trigger_laugh_response(self, humor_level: int):
        """
        Trigger physical laugh response based on humor level.
        
        Args:
            humor_level: How funny (1-10)
        """
        try:
            # Animation - bigger laugh for funnier jokes
            if self.animation:
                if humor_level >= 8:
                    # Big laugh - happy wiggle
                    self.animation.dispatch("play", "happy_wiggle")
                    logger.debug("Playing happy_wiggle animation")
                elif humor_level >= 6:
                    # Medium laugh - nod
                    self.animation.dispatch("play", "nod")
                    logger.debug("Playing nod animation")
                else:
                    # Small chuckle - subtle nod
                    self.animation.dispatch("play", "nod")
                    logger.debug("Playing nod animation (chuckle)")
            
            # RGB - warm flash
            if self.rgb:
                # Yellow/orange burst for warmth
                if humor_level >= 8:
                    color = [255, 200, 50]  # Bright warm yellow
                else:
                    color = [255, 150, 50]  # Warmer orange
                
                self.rgb.dispatch("animation", {
                    "name": "burst",
                    "color": color
                })
                logger.debug(f"Playing RGB burst with color {color}")
            
            # Sound - play a cheerful sound (using existing assets)
            if self.audio:
                # Use existing positive sound as laugh substitute
                sounds_to_try = [
                    "effects/scifi-positivedigitization",
                    "effects/scifi-pointcollected",
                    "effects/scifi-success"
                ]
                for sound in sounds_to_try:
                    try:
                        self.audio.play_sound(sound)
                        logger.debug(f"Playing sound: {sound}")
                        break
                    except:
                        continue
                        
        except Exception as e:
            logger.error(f"Error triggering laugh response: {e}")

    def enable(self):
        """Enable joke detection."""
        self._enabled = True
        logger.info("JokeHandler enabled")

    def disable(self):
        """Disable joke detection."""
        self._enabled = False
        logger.info("JokeHandler disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled
