#!/usr/bin/env python3
"""
Test script for joke detection feature.

Run: python test_joke_detection.py
"""

import asyncio
import sys

# Test phrases - mix of jokes and non-jokes
TEST_PHRASES = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "I'm going to the store to buy groceries.",
    "What do you call a fake noodle? An impasta!",
    "The weather is nice today.",
    "Why did the scarecrow win an award? He was outstanding in his field!",
    "Please turn on the lights.",
    "I used to hate facial hair, but then it grew on me.",
]


async def test_gemini_only():
    """Test just the Gemini joke detection."""
    print("\n" + "="*60)
    print("TEST 1: Gemini Joke Detection (standalone)")
    print("="*60 + "\n")
    
    try:
        from lelamp.service.gemini import GeminiService
    except ImportError:
        print("❌ Could not import GeminiService")
        print("   Run: pip install google-generativeai")
        return False
    
    # Load API key from config
    try:
        import yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        api_key = config.get("gemini", {}).get("api_key", "")
        if not api_key:
            print("❌ No Gemini API key in config.yaml")
            return False
    except Exception as e:
        print(f"❌ Could not load config: {e}")
        return False
    
    # Initialize service
    try:
        gemini = GeminiService(api_key=api_key)
        gemini.set_cooldown(0)  # Disable cooldown for testing
        print("✅ GeminiService initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize GeminiService: {e}")
        return False
    
    # Test each phrase
    for phrase in TEST_PHRASES:
        print(f"📝 Testing: \"{phrase[:50]}...\"" if len(phrase) > 50 else f"📝 Testing: \"{phrase}\"")
        
        try:
            result = await gemini.detect_joke(phrase)
            
            if result.get("is_joke"):
                print(f"   😂 JOKE DETECTED!")
                print(f"      Type: {result.get('joke_type')}")
                print(f"      Humor level: {result.get('humor_level')}/10")
            else:
                print(f"   😐 Not a joke")
            print()
            
            # Small delay between API calls
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    return True


async def test_joke_handler():
    """Test the full JokeHandler (without hardware)."""
    print("\n" + "="*60)
    print("TEST 2: JokeHandler (mock hardware)")
    print("="*60 + "\n")
    
    try:
        from lelamp.service.gemini import GeminiService
        from lelamp.service.joke_handler import JokeHandler
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Load config
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    api_key = config.get("gemini", {}).get("api_key", "")
    
    # Create mock services
    class MockService:
        def dispatch(self, action, data):
            print(f"      🎬 Mock dispatch: {action} -> {data}")
        
        def play_sound(self, sound):
            print(f"      🔊 Mock play sound: {sound}")
    
    mock = MockService()
    
    # Initialize
    gemini = GeminiService(api_key=api_key)
    gemini.set_cooldown(0)
    
    handler = JokeHandler(
        gemini_service=gemini,
        backboard_service=None,  # No Backboard for this test
        animation_service=mock,
        audio_service=mock,
        rgb_service=mock,
        min_humor_level=5
    )
    
    print("✅ JokeHandler initialized (with mock hardware)\n")
    
    # Test a joke
    joke = "Why don't eggs tell jokes? They'd crack each other up!"
    print(f"📝 Testing: \"{joke}\"\n")
    
    result = await handler.process_text(joke)
    
    print(f"\n📊 Result:")
    print(f"   Is joke: {result.get('is_joke')}")
    print(f"   Humor level: {result.get('humor_level')}")
    print(f"   Joke type: {result.get('joke_type')}")
    print(f"   Actions taken: {result.get('actions')}")
    
    return True


async def main():
    print("\n🤖 LeLamp Joke Detection Test Suite")
    print("="*60)
    
    # Test 1: Gemini only
    success1 = await test_gemini_only()
    
    if not success1:
        print("\n❌ Gemini test failed. Fix issues before continuing.")
        sys.exit(1)
    
    # Test 2: Full handler
    success2 = await test_joke_handler()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"  Gemini Detection: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  JokeHandler:      {'✅ PASS' if success2 else '❌ FAIL'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
