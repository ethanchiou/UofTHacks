"""
Backboard.io Memory Service for LeLamp.

Provides persistent memory storage and retrieval using Backboard SDK.
Stores user preferences, joke tendencies, and conversation context.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Try to import Backboard SDK
try:
    from backboard import BackboardClient
    BACKBOARD_AVAILABLE = True
except ImportError:
    BACKBOARD_AVAILABLE = False
    logger.warning("backboard-sdk not installed. Run: pip install backboard-sdk")


class BackboardService:
    """
    Backboard.io memory service for persistent user context.
    
    Features:
    - Store user preferences and tendencies
    - Track joke reactions and humor preferences
    - Retrieve relevant memories for context
    - Cross-session memory persistence
    """

    def __init__(self, api_key: str = None):
        """
        Initialize Backboard service.
        
        Args:
            api_key: Backboard.io API key (or uses BACKBOARD_API_KEY env var)
        """
        import os
        
        if not BACKBOARD_AVAILABLE:
            raise ImportError("backboard-sdk not installed")
        
        # Get API key from param or environment
        self.api_key = api_key or os.getenv("BACKBOARD_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set BACKBOARD_API_KEY in .env")
        
        self.client = BackboardClient(api_key=self.api_key)
        self.assistant = None
        self.assistant_id = None
        self.thread = None
        self.thread_id = None
        self._initialized = False
        
        # Cumulative humor stats (local cache, synced to Backboard)
        self._humor_stats = {
            "total_jokes": 0,
            "total_humor_points": 0,
            "joke_types": {},  # {"pun": 3, "wordplay": 2}
        }
        
        logger.info("BackboardService created (not yet initialized)")

    async def initialize(self, assistant_id: Optional[str] = None):
        """
        Initialize or reconnect to Backboard assistant.
        
        Args:
            assistant_id: Existing assistant ID to reuse, or None to create new
        """
        try:
            if assistant_id:
                # Reuse existing assistant
                self.assistant_id = assistant_id
                logger.info(f"Reusing existing assistant: {assistant_id}")
            else:
                # Create new assistant
                self.assistant = await self.client.create_assistant(
                    name="LeLamp Memory",
                    system_prompt="""You are a memory system for LeLamp, a robot lamp.
You remember user preferences, humor tendencies, and important context.
When asked what you remember, provide relevant stored information concisely."""
                )
                self.assistant_id = self.assistant.assistant_id
                logger.info(f"Created new assistant: {self.assistant_id}")
            
            # Create a new thread for this session
            self.thread = await self.client.create_thread(self.assistant_id)
            self.thread_id = self.thread.thread_id
            
            self._initialized = True
            logger.info(f"BackboardService initialized (thread: {self.thread_id})")
            
            return self.assistant_id
            
        except Exception as e:
            logger.error(f"Failed to initialize Backboard: {e}")
            self._initialized = False
            raise

    async def store_joke_reaction(self, joke_text: str, joke_type: str, humor_level: int):
        """
        Store that user told or reacted to a joke and update cumulative stats.
        
        Args:
            joke_text: The joke content
            joke_type: Type of joke (pun, sarcasm, etc.)
            humor_level: Humor rating 1-10
            
        Returns:
            dict with updated cumulative stats
        """
        # Update local cumulative stats
        self._humor_stats["total_jokes"] += 1
        self._humor_stats["total_humor_points"] += humor_level
        self._humor_stats["joke_types"][joke_type] = self._humor_stats["joke_types"].get(joke_type, 0) + 1
        
        # Calculate derived stats
        avg_humor = self._humor_stats["total_humor_points"] / self._humor_stats["total_jokes"]
        favorite_type = max(self._humor_stats["joke_types"], key=self._humor_stats["joke_types"].get)
        
        stats_summary = {
            "total_jokes": self._humor_stats["total_jokes"],
            "average_humor": round(avg_humor, 1),
            "favorite_joke_type": favorite_type,
            "joke_type_counts": self._humor_stats["joke_types"].copy()
        }
        
        if not self._initialized:
            logger.warning("BackboardService not initialized, stats updated locally only")
            return stats_summary
        
        try:
            content = f"""User interaction - JOKE DETECTED:
- Joke type: {joke_type}
- Humor level: {humor_level}/10
- Content snippet: "{joke_text[:100]}..."

CUMULATIVE HUMOR PROFILE UPDATE:
- Total jokes told: {stats_summary['total_jokes']}
- Average humor score: {stats_summary['average_humor']}/10
- Favorite joke type: {stats_summary['favorite_joke_type']}
- Joke type breakdown: {stats_summary['joke_type_counts']}

Remember: This user enjoys {joke_type} humor and has an average humor score of {stats_summary['average_humor']}/10."""

            await self.client.add_message(
                thread_id=self.thread_id,
                content=content,
                memory="Auto",  # Automatically persist to memory
                stream=False
            )
            
            logger.info(f"Stored joke: type={joke_type}, level={humor_level}, avg={stats_summary['average_humor']}")
            
            return stats_summary
            
        except Exception as e:
            logger.error(f"Failed to store joke reaction: {e}")
            return stats_summary
    
    def get_humor_stats(self) -> dict:
        """Get current cumulative humor statistics."""
        if self._humor_stats["total_jokes"] == 0:
            return {
                "total_jokes": 0,
                "average_humor": 0,
                "favorite_joke_type": None,
                "joke_type_counts": {}
            }
        
        avg_humor = self._humor_stats["total_humor_points"] / self._humor_stats["total_jokes"]
        favorite_type = max(self._humor_stats["joke_types"], key=self._humor_stats["joke_types"].get)
        
        return {
            "total_jokes": self._humor_stats["total_jokes"],
            "average_humor": round(avg_humor, 1),
            "favorite_joke_type": favorite_type,
            "joke_type_counts": self._humor_stats["joke_types"].copy()
        }

    async def store_preference(self, preference_key: str, preference_value: Any):
        """
        Store a user preference.
        
        Args:
            preference_key: Name of preference (e.g., "favorite_color")
            preference_value: Value of preference
        """
        if not self._initialized:
            logger.warning("BackboardService not initialized, skipping store")
            return
        
        try:
            content = f"""User preference learned:
- {preference_key}: {preference_value}
Remember this for future interactions."""

            await self.client.add_message(
                thread_id=self.thread_id,
                content=content,
                memory="Auto",
                stream=False
            )
            
            logger.info(f"Stored preference: {preference_key}={preference_value}")
            
        except Exception as e:
            logger.error(f"Failed to store preference: {e}")

    async def get_user_context(self, query: str = "What do you remember about this user?") -> str:
        """
        Retrieve relevant memories about the user.
        
        Args:
            query: What to ask the memory system
            
        Returns:
            String with relevant remembered context
        """
        if not self._initialized:
            return ""
        
        try:
            response = await self.client.add_message(
                thread_id=self.thread_id,
                content=query,
                memory="Auto",  # Retrieves relevant memories
                stream=False
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
            return ""

    async def get_humor_preferences(self) -> str:
        """Get user's humor preferences from memory."""
        return await self.get_user_context(
            "What types of jokes and humor does this user enjoy? Be concise."
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized
