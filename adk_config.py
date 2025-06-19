"""ADK Configuration for Blender AI Agent Studio"""

from google.genai.adk import Agent
from agents.scene_director import SceneDirectorAgent, AgentConfig


def create_scene_director_agent() -> Agent:
    """Create and configure the Scene Director Agent for ADK"""
    config = AgentConfig(
        name="scene_director",
        model="gemini-2.0-flash-exp",
        temperature=0.8,
        max_tokens=4096
    )
    
    return SceneDirectorAgent(config)


def create_blender_studio_agents() -> dict:
    """Create all Blender Studio agents for ADK"""
    agents = {}
    
    # Scene Director - Main orchestrator
    agents["scene_director"] = create_scene_director_agent()
    
    # Add other agents as they are converted to ADK format
    # agents["environment_builder"] = create_environment_builder_agent()
    # agents["lighting_designer"] = create_lighting_designer_agent()
    # agents["asset_placement"] = create_asset_placement_agent()
    # agents["effects_coordinator"] = create_effects_coordinator_agent()
    
    return agents


# Main agent for ADK Web UI
def get_main_agent() -> Agent:
    """Get the main agent for ADK Web UI"""
    return create_scene_director_agent()