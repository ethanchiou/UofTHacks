#!/usr/bin/env python3
"""
Test script for joke detection feature.

Run: python test_joke_detection.py
"""

import asyncio
import os
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


async def test_joke_detection():
    """Test the Gemini joke detection with real API."""
    print("\n" + "="*60)
    print("LeLamp Joke Detection Test")
    print("="*60 + "\n")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] No GEMINI_API_KEY found in .env")
        print("Create a .env file with: GEMINI_API_KEY=your_key")
        return False
    
    print(f"[OK] API key found: {api_key[:10]}...")
    
    # Import and initialize
    try:
        from lelamp.service.gemini import GeminiService
        gemini = GeminiService(api_key=api_key)
        gemini.set_cooldown(0)  # No cooldown for testing
        print("[OK] GeminiService initialized\n")
    except Exception as e:
        print(f"[ERROR] Failed to init GeminiService: {e}")
        return False
    
    # Test jokes
    test_cases = [
        ("Why don't scientists trust atoms? Because they make up everything!", True),
        ("I need to buy groceries tomorrow.", False),
        ("What do you call a fake noodle? An impasta!", True),
        ("The meeting is at 3pm.", False),
        ("I used to hate facial hair, but then it grew on me.", True),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_joke in test_cases:
        print(f"Input: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        
        try:
            result = await gemini.detect_joke(text)
            is_joke = result.get("is_joke", False)
            humor = result.get("humor_level", 0)
            jtype = result.get("joke_type", "none")
            
            status = "PASS" if is_joke == expected_joke else "FAIL"
            if status == "PASS":
                passed += 1
            else:
                failed += 1
            
            if is_joke:
                print(f"  -> JOKE! Type: {jtype}, Humor: {humor}/10 [{status}]")
            else:
                print(f"  -> Not a joke [{status}]")
            print()
            
            await asyncio.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            failed += 1
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


async def main():
    success = await test_joke_detection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
