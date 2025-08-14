"""
Client test script for the LLM Serving API
"""
import asyncio
import aiohttp
import json
import time
from typing import Dict, Any


class LLMClient:
    """Simple client for testing the LLM Serving API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base}/health") as response:
                return await response.json()
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.api_base}/model-info") as response:
                return await response.json()
    
    async def load_model(self) -> Dict[str, Any]:
        """Explicitly load the model"""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.api_base}/load-model") as response:
                return await response.json()
    
    async def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1
    ) -> Dict[str, Any]:
        """Generate text using the API"""
        
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                return await response.json()


async def run_tests():
    """Run comprehensive tests of the API"""
    
    client = LLMClient()
    
    print("🚀 Starting LLM API Tests")
    print("=" * 50)
    
    try:
        # Test 1: Health Check
        print("1. Testing health check...")
        health = await client.health_check()
        print(f"   ✅ Health: {health['status']}")
        print(f"   ⏱️ Uptime: {health['uptime']:.2f}s")
        
        # Test 2: Model Info
        print("\n2. Testing model info...")
        model_info = await client.get_model_info()
        print(f"   📦 Model: {model_info['model_name']}")
        print(f"   💾 Size: {model_info['model_size']}")
        print(f"   🏭 Device: {model_info['device']}")
        print(f"   ✅ Loaded: {model_info['loaded']}")
        
        # Test 3: Load Model (if not already loaded)
        if not model_info['loaded']:
            print("\n3. Loading model...")
            load_result = await client.load_model()
            print(f"   ✅ {load_result['message']}")
            print(f"   ⏱️ Load time: {load_result['load_time']:.2f}s")
        
        # Test 4: Simple Text Generation
        print("\n4. Testing text generation...")
        prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a magical forest",
            "The benefits of renewable energy include"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n   Test 4.{i}: '{prompt}'")
            start_time = time.time()
            
            result = await client.generate_text(
                prompt=prompt,
                max_length=50,
                temperature=0.8
            )
            
            end_time = time.time()
            
            print(f"   📝 Generated: {result['generated_text'][0]}")
            print(f"   ⏱️ Time: {result['generation_time']:.2f}s")
            print(f"   🌡️ Temperature: {result['parameters']['temperature']}")
        
        # Test 5: Multiple Sequences
        print("\n5. Testing multiple sequences...")
        result = await client.generate_text(
            prompt="The three most important things in life are",
            max_length=30,
            num_return_sequences=3,
            temperature=0.9
        )
        
        for i, text in enumerate(result['generated_text'], 1):
            print(f"   Sequence {i}: {text}")
        
        # Test 6: Parameter Variations
        print("\n6. Testing parameter variations...")
        
        # High temperature (more creative)
        result_creative = await client.generate_text(
            prompt="The robot decided to",
            max_length=40,
            temperature=1.2
        )
        print(f"   🎨 Creative (temp=1.2): {result_creative['generated_text'][0]}")
        
        # Low temperature (more deterministic)
        result_conservative = await client.generate_text(
            prompt="The robot decided to",
            max_length=40,
            temperature=0.3
        )
        print(f"   🎯 Conservative (temp=0.3): {result_conservative['generated_text'][0]}")
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False
    
    return True


async def interactive_mode():
    """Interactive mode for manual testing"""
    
    client = LLMClient()
    
    print("\n🎮 Interactive Mode")
    print("Type 'exit' to quit, 'help' for commands")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nPrompt: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  help - Show this help")
                print("  exit - Quit interactive mode")
                print("  health - Check API health")
                print("  info - Get model info")
                print("  Or just type a prompt for text generation")
                continue
            elif user_input.lower() == 'health':
                health = await client.health_check()
                print(f"Health: {health}")
                continue
            elif user_input.lower() == 'info':
                info = await client.get_model_info()
                print(f"Model Info: {info}")
                continue
            elif not user_input:
                continue
            
            # Generate text
            result = await client.generate_text(
                prompt=user_input,
                max_length=100
            )
            
            print(f"\n📝 Generated: {result['generated_text'][0]}")
            print(f"⏱️ Time: {result['generation_time']:.2f}s")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n👋 Goodbye!")


async def main():
    """Main function"""
    
    print("LLM Serving API Test Client")
    print("=" * 30)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_mode()
    else:
        success = await run_tests()
        
        if success:
            print("\n🎮 Run with --interactive flag for interactive mode")
        
        return success


if __name__ == "__main__":
    # Run the async main function
    result = asyncio.run(main())