#!/usr/bin/env python3
"""Main entry point for ADK Web UI integration"""

import asyncio
import os
from pathlib import Path

from adk_config import get_main_agent


def setup_environment():
    """Setup environment variables and paths"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Ensure Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key in a .env file or environment variable")
        exit(1)
    
    # Set up project paths
    project_root = Path(__file__).parent
    os.environ["BLENDER_ASSETS_PATH"] = str(project_root / "assets")
    os.environ["BLENDER_OUTPUT_PATH"] = str(project_root / "output")
    
    # Create output directory if it doesn't exist
    (project_root / "output").mkdir(exist_ok=True)


def main():
    """Main entry point for ADK integration"""
    print("🎬 Blender AI Agent Studio - ADK Integration")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Get the main agent
    agent = get_main_agent()
    
    print(f"✅ Agent '{agent.name}' initialized successfully")
    print("\nAvailable tools:")
    
    # List available tools
    for tool_name in dir(agent):
        if hasattr(getattr(agent, tool_name), '_is_tool'):
            tool_func = getattr(agent, tool_name)
            print(f"  • {tool_name}: {tool_func.__doc__ or 'No description'}")
    
    print("\n🌐 To start ADK Web UI, run:")
    print("   adk web")
    print("\n📝 The agent will be available in the ADK Web interface")
    print("   at http://localhost:8000")
    
    return agent


if __name__ == "__main__":
    agent = main()