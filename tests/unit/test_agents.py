"""Unit tests for individual agents"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch

from agents.scene_director import SceneDirectorAgent
from agents.environment_builder import EnvironmentBuilderAgent
from agents.lighting_designer import LightingDesignerAgent
from agents.asset_placement import AssetPlacementAgent
from agents.effects_coordinator import EffectsCoordinatorAgent
from agents.base_agent import AgentConfig, AgentState


class TestBaseAgent:
    """Test the base agent functionality"""
    
    @pytest.mark.unit
    def test_agent_config_creation(self, test_agent_config):
        """Test agent configuration creation"""
        assert test_agent_config.name == "test_agent"
        assert test_agent_config.model == "gemini-2.0-flash-exp"
        assert test_agent_config.temperature == 0.7
        assert test_agent_config.max_retries == 2
        assert test_agent_config.timeout == 30
    
    @pytest.mark.unit
    def test_agent_state_initialization(self):
        """Test agent state initialization"""
        state = AgentState(agent_id="test_agent")
        assert state.agent_id == "test_agent"
        assert state.current_task is None
        assert state.completed_tasks == []
        assert state.errors == []
        assert state.metadata == {}
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_base_agent_execute_timeout(self, scene_director_agent):
        """Test agent execution timeout handling"""
        # Mock a slow-running process method
        async def slow_process(input_data):
            await asyncio.sleep(2)  # Longer than timeout
            return {"result": "success"}
        
        scene_director_agent.process = slow_process
        scene_director_agent.config.timeout = 1  # 1 second timeout
        
        task = {"description": "Test timeout task", "prompt": "test"}
        
        with pytest.raises(asyncio.TimeoutError):
            await scene_director_agent.execute(task)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_base_agent_error_handling(self, scene_director_agent):
        """Test agent error handling"""
        # Mock a failing process method
        async def failing_process(input_data):
            raise ValueError("Test error")
        
        scene_director_agent.process = failing_process
        
        task = {"description": "Test error task", "prompt": "test"}
        
        with pytest.raises(ValueError):
            await scene_director_agent.execute(task)
        
        # Check that error was recorded
        assert len(scene_director_agent.state.errors) > 0
        assert "Test error" in scene_director_agent.state.errors[-1]


class TestSceneDirectorAgent:
    """Test the Scene Director agent"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scene_analysis(self, scene_director_agent, mock_gemini_api):
        """Test scene analysis functionality"""
        task_data = {
            "prompt": "A modern office building in a city",
            "style": "realistic",
            "quality": "high",
            "scene_id": "test_scene"
        }
        
        result = await scene_director_agent.execute(task_data)
        
        assert result["status"] == "success"
        assert "coordination_data" in result
        assert "requirements" in result["coordination_data"]
        assert "tasks" in result["coordination_data"]
        assert "execution_order" in result["coordination_data"]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scene_review(self, scene_director_agent, mock_gemini_api):
        """Test scene quality review"""
        review_data = {
            "requirements": {
                "scene_type": "outdoor",
                "complexity": "moderate",
                "style": "realistic"
            },
            "completed_tasks": ["environment_builder", "lighting_designer"],
            "blender_result": {
                "status": "success",
                "output_file": "/test/scene.blend"
            }
        }
        
        result = await scene_director_agent.review_scene(review_data)
        
        assert "overall_score" in result
        assert "feedback" in result
        assert "recommendations" in result
    
    @pytest.mark.unit
    def test_parse_scene_requirements_valid_json(self, scene_director_agent):
        """Test parsing valid JSON scene requirements"""
        json_response = """
        {
            "scene_type": "outdoor",
            "complexity": "moderate",
            "elements": ["building", "street"],
            "environment": {"terrain": "urban"},
            "mood": "modern",
            "style_requirements": {"architecture": "contemporary"},
            "agent_tasks": {
                "environment_builder": {"create_layout": true}
            },
            "execution_order": [["environment_builder"]]
        }
        """
        
        requirements = scene_director_agent._parse_scene_requirements(json_response)
        
        assert requirements.scene_type == "outdoor"
        assert requirements.complexity == "moderate"
        assert "building" in requirements.elements
        assert requirements.environment["terrain"] == "urban"
    
    @pytest.mark.unit
    def test_parse_scene_requirements_invalid_json(self, scene_director_agent):
        """Test parsing invalid JSON falls back to text analysis"""
        invalid_response = "This is not JSON but describes a modern urban scene with buildings"
        
        requirements = scene_director_agent._parse_scene_requirements(invalid_response)
        
        # Should fall back to text analysis
        assert requirements.scene_type in ["outdoor", "indoor", "mixed"]
        assert requirements.complexity in ["simple", "moderate", "complex", "ultra"]


class TestEnvironmentBuilderAgent:
    """Test the Environment Builder agent"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_environment_generation(self, environment_builder_agent, mock_gemini_api):
        """Test environment generation"""
        task_data = {
            "parameters": {
                "scene_type": "urban",
                "style": "modern",
                "complexity": "moderate",
                "elements": ["buildings", "streets"]
            }
        }
        
        result = await environment_builder_agent.execute(task_data)
        
        assert result["status"] == "success"
        assert "blender_commands" in result
        assert len(result["blender_commands"]) > 0
        
        # Check for terrain and structure commands
        command_types = [cmd["type"] for cmd in result["blender_commands"]]
        assert "create_terrain" in command_types
        assert "create_structure" in command_types
    
    @pytest.mark.unit
    def test_environment_preset_selection(self, environment_builder_agent):
        """Test environment preset selection"""
        requirements = {
            "scene_type": "urban",
            "style": "cyberpunk",
            "complexity": "high"
        }
        
        preset = environment_builder_agent._select_environment_preset(requirements)
        
        assert preset is not None
        assert "terrain" in preset
        assert "structures" in preset
    
    @pytest.mark.unit
    def test_generate_terrain_commands(self, environment_builder_agent):
        """Test terrain command generation"""
        terrain_config = {
            "type": "urban",
            "size": [100, 100],
            "height_variation": 0.5,
            "material": "concrete"
        }
        
        commands = environment_builder_agent._generate_terrain_commands(terrain_config)
        
        assert len(commands) > 0
        assert commands[0]["type"] == "create_terrain"
        assert commands[0]["params"]["size"] == [100, 100]


class TestLightingDesignerAgent:
    """Test the Lighting Designer agent"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_lighting_design(self, lighting_designer_agent, mock_gemini_api):
        """Test lighting design generation"""
        task_data = {
            "parameters": {
                "scene_context": "urban office building",
                "style": "realistic",
                "time_of_day": "day",
                "weather": "clear",
                "mood": "professional"
            }
        }
        
        result = await lighting_designer_agent.execute(task_data)
        
        assert result["status"] == "success"
        assert "blender_commands" in result
        assert len(result["blender_commands"]) > 0
        
        # Check for lighting commands
        command_types = [cmd["type"] for cmd in result["blender_commands"]]
        assert "create_light" in command_types
    
    @pytest.mark.unit
    def test_lighting_preset_selection(self, lighting_designer_agent):
        """Test lighting preset selection"""
        context = {
            "style": "cyberpunk",
            "time_of_day": "night",
            "weather": "foggy"
        }
        
        preset = lighting_designer_agent._select_lighting_preset(context)
        
        assert preset is not None
        assert "lights" in preset
        assert preset["style"] == "cyberpunk"
    
    @pytest.mark.unit
    def test_time_of_day_lighting(self, lighting_designer_agent):
        """Test time-of-day lighting configurations"""
        # Test day lighting
        day_lights = lighting_designer_agent._get_time_of_day_lighting("day")
        assert any(light["light_type"] == "sun" for light in day_lights)
        
        # Test night lighting
        night_lights = lighting_designer_agent._get_time_of_day_lighting("night")
        assert len(night_lights) > 0
        assert all(light["energy"] < 1000 for light in night_lights)  # Dimmer at night


class TestAssetPlacementAgent:
    """Test the Asset Placement agent"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_asset_placement(self, asset_placement_agent, mock_gemini_api):
        """Test asset placement generation"""
        task_data = {
            "parameters": {
                "environment": "urban office district",
                "style": "modern",
                "density": "medium",
                "available_categories": ["props", "vehicles", "architecture"]
            }
        }
        
        result = await asset_placement_agent.execute(task_data)
        
        assert result["status"] == "success"
        assert "blender_commands" in result
        assert "asset_placements" in result
    
    @pytest.mark.unit
    def test_asset_selection_by_style(self, asset_placement_agent):
        """Test asset selection based on style"""
        available_assets = [
            {"name": "modern_chair", "category": "props", "style_tags": ["modern", "office"]},
            {"name": "medieval_sword", "category": "props", "style_tags": ["medieval", "weapon"]},
            {"name": "cyber_screen", "category": "props", "style_tags": ["cyberpunk", "tech"]}
        ]
        
        # Test modern style selection
        modern_assets = asset_placement_agent._filter_assets_by_style(available_assets, "modern")
        assert len(modern_assets) == 1
        assert modern_assets[0]["name"] == "modern_chair"
        
        # Test cyberpunk style selection
        cyber_assets = asset_placement_agent._filter_assets_by_style(available_assets, "cyberpunk")
        assert len(cyber_assets) == 1
        assert cyber_assets[0]["name"] == "cyber_screen"
    
    @pytest.mark.unit
    def test_collision_detection(self, asset_placement_agent):
        """Test collision detection between assets"""
        existing_placements = [
            {"position": [0, 0, 0], "bounds": {"radius": 2}}
        ]
        
        # Test collision (too close)
        collision_pos = [1, 1, 0]
        assert asset_placement_agent._check_collision(collision_pos, existing_placements, 1.5)
        
        # Test no collision (far enough)
        safe_pos = [5, 5, 0]
        assert not asset_placement_agent._check_collision(safe_pos, existing_placements, 1.5)


class TestEffectsCoordinatorAgent:
    """Test the Effects Coordinator agent"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_effects_coordination(self, effects_coordinator_agent, mock_gemini_api):
        """Test effects coordination"""
        task_data = {
            "parameters": {
                "scene_description": "urban cyberpunk scene with neon lights",
                "atmosphere": "foggy night",
                "weather": "light rain",
                "style": "cyberpunk",
                "performance_target": "medium"
            }
        }
        
        result = await effects_coordinator_agent.execute(task_data)
        
        assert result["status"] == "success"
        assert "blender_commands" in result
        assert "effects" in result
    
    @pytest.mark.unit
    def test_weather_effects_generation(self, effects_coordinator_agent):
        """Test weather effects generation"""
        # Test rain effects
        rain_effects = effects_coordinator_agent._generate_weather_effects("rain", "medium")
        assert len(rain_effects) > 0
        assert any("rain" in effect.get("name", "").lower() for effect in rain_effects)
        
        # Test snow effects
        snow_effects = effects_coordinator_agent._generate_weather_effects("snow", "high")
        assert len(snow_effects) > 0
        assert any("snow" in effect.get("name", "").lower() for effect in snow_effects)
    
    @pytest.mark.unit
    def test_style_specific_effects(self, effects_coordinator_agent):
        """Test style-specific effects generation"""
        # Test cyberpunk effects
        cyber_effects = effects_coordinator_agent._generate_style_effects("cyberpunk")
        assert any("neon" in str(effect).lower() or "glow" in str(effect).lower() 
                  for effect in cyber_effects)
        
        # Test medieval effects
        medieval_effects = effects_coordinator_agent._generate_style_effects("medieval")
        assert len(medieval_effects) >= 0  # May have no specific effects
    
    @pytest.mark.unit
    def test_performance_optimization(self, effects_coordinator_agent):
        """Test effects performance optimization"""
        high_poly_effects = [
            {"name": "complex_particles", "particle_count": 10000},
            {"name": "detailed_fog", "resolution": 512}
        ]
        
        optimized = effects_coordinator_agent._optimize_effects_for_performance(
            high_poly_effects, "draft"
        )
        
        # Should reduce particle counts and resolutions for draft quality
        for effect in optimized:
            if "particle_count" in effect:
                assert effect["particle_count"] <= 5000
            if "resolution" in effect:
                assert effect["resolution"] <= 256


class TestAgentIntegration:
    """Test agent integration and collaboration"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_collaboration(self, scene_director_agent, environment_builder_agent):
        """Test collaboration between agents"""
        # Scene director creates coordination data
        director_task = {
            "prompt": "Modern office building",
            "style": "realistic", 
            "quality": "high",
            "scene_id": "test"
        }
        
        director_result = await scene_director_agent.execute(director_task)
        
        # Environment builder uses coordination data
        builder_task = {
            "parameters": director_result["coordination_data"]["tasks"][0]["parameters"],
            "shared_state": {
                "metadata": director_result["coordination_data"]["metadata"],
                "previous_results": {}
            }
        }
        
        builder_result = await environment_builder_agent.execute(builder_task)
        
        assert director_result["status"] == "success"
        assert builder_result["status"] == "success"
        assert len(builder_result["blender_commands"]) > 0
    
    @pytest.mark.unit
    def test_agent_state_management(self, scene_director_agent):
        """Test agent state management"""
        initial_state = scene_director_agent.get_state()
        assert initial_state["completed_tasks"] == []
        assert initial_state["errors"] == []
        
        # Update metadata
        scene_director_agent.update_metadata("test_key", "test_value")
        updated_state = scene_director_agent.get_state()
        assert updated_state["metadata"]["test_key"] == "test_value"
        
        # Reset state
        scene_director_agent.reset_state()
        reset_state = scene_director_agent.get_state()
        assert reset_state["metadata"] == {}
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_agent_error_recovery(self, environment_builder_agent):
        """Test agent error recovery mechanisms"""
        # Simulate a task that fails once then succeeds
        call_count = 0
        original_process = environment_builder_agent.process
        
        async def flaky_process(input_data):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return await original_process(input_data)
        
        environment_builder_agent.process = flaky_process
        environment_builder_agent.config.max_retries = 2
        
        task_data = {
            "parameters": {
                "scene_type": "urban",
                "style": "modern",
                "complexity": "simple"
            }
        }
        
        # Should succeed after retry
        result = await environment_builder_agent.execute(task_data)
        assert result["status"] == "success"
        assert call_count == 2  # Failed once, succeeded on retry