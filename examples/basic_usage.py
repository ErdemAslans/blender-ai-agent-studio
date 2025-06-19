"""Basic usage examples for Blender AI Agent Studio"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from main import BlenderAIStudio


async def example_basic_scene():
    """Generate a basic scene"""
    studio = BlenderAIStudio()
    
    result = await studio.generate_scene(
        prompt="Create a simple scene with a red cube on a grassy terrain",
        style="realistic",
        quality="draft"
    )
    
    print(f"Basic scene result: {result['status']}")
    if result['status'] == 'success':
        print(f"Output file: {result['output_file']}")
    
    return result


async def example_cyberpunk_city():
    """Generate a cyberpunk city scene"""
    studio = BlenderAIStudio()
    
    result = await studio.generate_scene(
        prompt="Create a cyberpunk city street at night with neon signs, rain, and futuristic vehicles",
        style="cyberpunk", 
        quality="high"
    )
    
    print(f"Cyberpunk scene result: {result['status']}")
    if result['status'] == 'success':
        print(f"Execution time: {result['execution_time']:.2f}s")
        print(f"Output file: {result['output_file']}")
        
        # Also render an image
        render_result = await studio.render_scene(result['output_file'])
        if render_result['status'] == 'success':
            print(f"Rendered image: {render_result['output_image']}")
    
    return result


async def example_medieval_castle():
    """Generate a medieval castle scene"""
    studio = BlenderAIStudio()
    
    result = await studio.generate_scene(
        prompt="A medieval castle on a hilltop surrounded by a misty forest with warm torchlight",
        style="medieval",
        quality="high"
    )
    
    print(f"Medieval scene result: {result['status']}")
    return result


async def example_batch_generation():
    """Generate multiple scenes in batch"""
    studio = BlenderAIStudio()
    
    prompts = [
        ("A modern coffee shop interior with warm lighting", "realistic"),
        ("An alien planet landscape with strange rock formations", "scifi"),
        ("A post-apocalyptic abandoned city with overgrown vegetation", "post-apocalyptic")
    ]
    
    results = []
    
    for prompt, style in prompts:
        print(f"Generating: {prompt[:50]}...")
        
        result = await studio.generate_scene(
            prompt=prompt,
            style=style,
            quality="preview"  # Use preview for faster batch processing
        )
        
        results.append(result)
        print(f"  Status: {result['status']}")
        
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nBatch complete: {successful}/{len(results)} scenes generated successfully")
    
    return results


async def example_with_custom_elements():
    """Generate scene with specific elements"""
    studio = BlenderAIStudio()
    
    result = await studio.generate_scene(
        prompt="Create an urban park scene with benches, lamp posts, trees, and a fountain in the center",
        style="realistic",
        quality="high"
    )
    
    print(f"Custom elements scene result: {result['status']}")
    
    # Show which elements were detected and placed
    if result['status'] == 'success':
        asset_results = result.get('agent_results', {}).get('asset_placement', {})
        if asset_results:
            stats = asset_results.get('statistics', {})
            print(f"Assets placed: {stats}")
    
    return result


async def main():
    """Run all examples"""
    print("🎬 Blender AI Agent Studio - Examples")
    print("=" * 50)
    
    # Example 1: Basic scene
    print("\n1. Basic Scene Example")
    print("-" * 30)
    await example_basic_scene()
    
    # Example 2: Cyberpunk city
    print("\n2. Cyberpunk City Example")
    print("-" * 30)
    await example_cyberpunk_city()
    
    # Example 3: Medieval castle
    print("\n3. Medieval Castle Example")
    print("-" * 30)
    await example_medieval_castle()
    
    # Example 4: Custom elements
    print("\n4. Custom Elements Example")
    print("-" * 30)
    await example_with_custom_elements()
    
    # Example 5: Batch generation
    print("\n5. Batch Generation Example")
    print("-" * 30)
    await example_batch_generation()
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())