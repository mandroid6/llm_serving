"""
Standalone test script for direct model testing without API server
"""

import asyncio
import os
import sys
import time

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from app.core.logging import setup_logging
from app.models.model_manager import model_manager

# Setup logging
setup_logging()


async def test_standalone_model():
    """Test the model directly without the API server"""

    print("🤖 LLM Standalone Model Test")
    print("=" * 50)

    try:
        # Test 1: Model Info (before loading)
        print("1. Getting model info (before loading)...")
        info = model_manager.get_model_info()
        print(f"   📦 Model: {info['model_name']}")
        print(f"   🏭 Device: {info['device']}")
        print(f"   ✅ Loaded: {info['loaded']}")

        # Test 2: Load Model
        print("\n2. Loading model...")
        start_time = time.time()
        success = await model_manager.load_model()
        load_time = time.time() - start_time

        if not success:
            print("   ❌ Failed to load model")
            return False

        print(f"   ✅ Model loaded successfully in {load_time:.2f}s")

        # Test 3: Model Info (after loading)
        print("\n3. Getting model info (after loading)...")
        info = model_manager.get_model_info()
        print(f"   📦 Model: {info['model_name']}")
        print(f"   🔧 Type: {info['model_type']}")
        print(f"   💾 Size: {info['model_size']}")
        print(f"   🏭 Device: {info['device']}")
        print(f"   📏 Max Length: {info['max_length']}")
        print(f"   ✅ Loaded: {info['loaded']}")

        # Test 4: Simple Text Generation
        print("\n4. Testing text generation...")
        test_prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a magical forest",
            "The benefits of renewable energy include",
            "In the year 2050, technology will",
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n   Test 4.{i}: '{prompt}'")
            start_time = time.time()

            result = await model_manager.generate_text(
                prompt=prompt, max_length=50, temperature=0.8, top_p=0.9, top_k=50
            )

            generation_time = time.time() - start_time

            print(f"   📝 Generated: {result['generated_text'][0]}")
            print(f"   ⏱️ API Time: {result['generation_time']:.3f}s")
            print(f"   ⏱️ Total Time: {generation_time:.3f}s")

        # Test 5: Parameter Variations
        print("\n5. Testing parameter variations...")
        base_prompt = "The robot decided to"

        # High creativity
        result_creative = await model_manager.generate_text(
            prompt=base_prompt, max_length=40, temperature=1.2, top_p=0.95
        )
        print(f"   🎨 Creative (temp=1.2): {result_creative['generated_text'][0]}")

        # Conservative
        result_conservative = await model_manager.generate_text(
            prompt=base_prompt, max_length=40, temperature=0.3, top_p=0.7
        )
        print(
            f"   🎯 Conservative (temp=0.3): {result_conservative['generated_text'][0]}"
        )

        # Test 6: Multiple Sequences
        print("\n6. Testing multiple sequences...")
        result_multi = await model_manager.generate_text(
            prompt="The three most important things in life are",
            max_length=25,
            num_return_sequences=3,
            temperature=0.9,
        )

        for i, text in enumerate(result_multi["generated_text"], 1):
            print(f"   Sequence {i}: {text}")

        # Test 7: Performance Test
        print("\n7. Performance test (10 quick generations)...")
        performance_times = []

        for i in range(10):
            start_time = time.time()
            result = await model_manager.generate_text(
                prompt=f"Test {i+1}:", max_length=20, temperature=0.7
            )
            end_time = time.time()
            performance_times.append(end_time - start_time)

            if i == 0:  # Show first result
                print(f"   Example: {result['generated_text'][0]}")

        avg_time = sum(performance_times) / len(performance_times)
        min_time = min(performance_times)
        max_time = max(performance_times)

        print(f"   📊 Average time: {avg_time:.3f}s")
        print(f"   📊 Min time: {min_time:.3f}s")
        print(f"   📊 Max time: {max_time:.3f}s")

        print("\n" + "=" * 50)
        print("✅ All standalone tests completed successfully!")
        print(f"🎯 Model: {info['model_name']} ({info['model_size']})")
        print(f"🏭 Device: {info['device']}")
        print(f"⚡ Average generation: {avg_time:.3f}s")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


async def interactive_standalone():
    """Interactive mode for direct model testing"""

    print("\n🎮 Interactive Standalone Mode")
    print("Type 'exit' to quit, 'help' for commands")
    print("-" * 40)

    # Load model if not already loaded
    if not model_manager.is_loaded:
        print("Loading model...")
        success = await model_manager.load_model()
        if not success:
            print("❌ Failed to load model")
            return
        print("✅ Model loaded!")

    while True:
        try:
            user_input = input("\nPrompt: ").strip()

            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "help":
                print("\nCommands:")
                print("  help - Show this help")
                print("  exit - Quit interactive mode")
                print("  info - Get model info")
                print("  Or just type a prompt for text generation")
                continue
            elif user_input.lower() == "info":
                info = model_manager.get_model_info()
                print(f"Model Info: {info}")
                continue
            elif not user_input:
                continue

            # Generate text
            result = await model_manager.generate_text(
                prompt=user_input, max_length=100, temperature=0.8
            )

            print(f"\n📝 Generated: {result['generated_text'][0]}")
            print(f"⏱️ Time: {result['generation_time']:.3f}s")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    print("\n👋 Goodbye!")


async def main():
    """Main function"""

    print("LLM Standalone Model Tester")
    print("=" * 30)

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_standalone()
    else:
        success = await test_standalone_model()

        if success:
            print("\n🎮 Run with --interactive flag for interactive mode")
            print("💡 This test bypasses the API server and calls the model directly")

        return success


if __name__ == "__main__":
    # Run the async main function
    result = asyncio.run(main())
