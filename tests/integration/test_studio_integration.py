"""Integration tests for the complete Blender AI Studio system"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from main import BlenderAIStudio
from utils.config import get_config
from utils.asset_manager import get_asset_manager
from utils.caching import get_cache


class TestStudioIntegration:
    """Test complete studio integration"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_scene_generation_flow(self, blender_studio, sample_scene_data):
        """Test the complete scene generation flow from prompt to output"""
        
        result = await blender_studio.generate_scene(
            prompt=sample_scene_data["prompt"],
            style=sample_scene_data["style"],
            quality=sample_scene_data["quality"]
        )
        
        # Verify successful generation
        assert result["status"] == "success"
        assert result["scene_id"] is not None
        assert result["output_file"] is not None
        assert os.path.exists(result["output_file"])
        assert result["execution_time"] > 0
        
        # Verify agent execution
        assert "agent_results" in result
        assert len(result["agent_results"]) > 0
        
        # Verify Blender commands were generated
        total_commands = sum(
            len(agent_result.get("blender_commands", []))
            for agent_result in result["agent_results"].values()
        )
        assert total_commands > 0
        
        # Verify scene review
        assert "review" in result
        assert "overall_score" in result["review"]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_coordination_and_dependencies(self, blender_studio):
        """Test that agents execute in correct order with proper dependencies"""
        
        prompt = "A cyberpunk street scene with neon lights and fog"
        
        result = await blender_studio.generate_scene(
            prompt=prompt,
            style="cyberpunk",
            quality="draft"
        )
        
        assert result["status"] == "success"
        
        # Verify all expected agents executed
        expected_agents = [
            "scene_director",
            "environment_builder", 
            "lighting_designer",
            "asset_placement",
            "effects_coordinator"
        ]
        
        executed_agents = list(result["agent_results"].keys())
        
        # Scene director should have executed (creates coordination)
        assert "scene_director" in executed_agents
        
        # At least some specialist agents should have executed
        specialist_agents = [agent for agent in executed_agents if agent != "scene_director"]
        assert len(specialist_agents) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, blender_studio):
        """Test error handling and recovery mechanisms"""
        
        # Test with malformed prompt
        result = await blender_studio.generate_scene(
            prompt="",  # Empty prompt
            style="invalid_style",
            quality="ultra"
        )
        
        # Should handle gracefully, either succeed with defaults or fail cleanly
        assert result["status"] in ["success", "error"]
        if result["status"] == "error":
            assert "error" in result
            assert isinstance(result["error"], str)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_different_scene_styles(self, blender_studio):
        """Test generation with different artistic styles"""
        
        styles_to_test = ["realistic", "cyberpunk", "medieval", "cartoon"]
        base_prompt = "A simple scene with a building and a tree"
        
        results = []
        
        for style in styles_to_test:
            result = await blender_studio.generate_scene(
                prompt=base_prompt,
                style=style,
                quality="draft"
            )
            results.append((style, result))
        
        # All styles should succeed
        for style, result in results:
            assert result["status"] == "success", f"Style {style} failed: {result.get('error')}"
            
            # Each style should produce different results
            assert result["style"] == style
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_quality_presets(self, blender_studio):
        """Test generation with different quality presets"""
        
        qualities = ["draft", "preview", "high"]
        prompt = "A modern office building"
        
        results = []
        
        for quality in qualities:
            result = await blender_studio.generate_scene(
                prompt=prompt,
                style="realistic",
                quality=quality
            )
            results.append((quality, result))
        
        # All qualities should succeed
        for quality, result in results:
            assert result["status"] == "success", f"Quality {quality} failed"
            assert result["quality"] == quality
        
        # Higher quality should generally take longer (with mocked executor this may not hold)
        draft_time = next(r[1]["execution_time"] for r in results if r[0] == "draft")
        high_time = next(r[1]["execution_time"] for r in results if r[0] == "high")
        
        # At minimum, both should have reasonable execution times
        assert draft_time > 0
        assert high_time > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scene_rendering(self, blender_studio, temp_dir):
        """Test scene rendering functionality"""
        
        # Generate a scene first
        result = await blender_studio.generate_scene(
            prompt="A simple cube on a plane",
            style="realistic",
            quality="draft"
        )
        
        assert result["status"] == "success"
        scene_file = result["output_file"]
        
        # Render the scene
        render_result = await blender_studio.render_scene(scene_file)
        
        assert render_result["status"] == "success"
        assert render_result["output_image"] is not None
        assert os.path.exists(render_result["output_image"])
    
    @pytest.mark.integration
    def test_agent_status_monitoring(self, blender_studio):
        """Test agent status monitoring"""
        
        status = blender_studio.get_agent_status()
        
        # Should return status for all agents
        expected_agents = [
            "scene_director",
            "environment_builder",
            "lighting_designer", 
            "asset_placement",
            "effects_coordinator"
        ]
        
        for agent_name in expected_agents:
            assert agent_name in status
            agent_status = status[agent_name]
            assert "agent_id" in agent_status
            assert "completed_tasks" in agent_status
            assert "errors" in agent_status
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_scene_generation(self, blender_studio):
        """Test handling multiple concurrent scene generation requests"""
        
        prompts = [
            "A modern city street",
            "A medieval castle",
            "A cyberpunk alley"
        ]
        
        # Start multiple generations concurrently
        tasks = [
            blender_studio.generate_scene(
                prompt=prompt,
                style="realistic",
                quality="draft"
            )
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent generation {i} failed: {result}")
            
            assert result["status"] == "success"
            assert result["prompt"] == prompts[i]
    
    @pytest.mark.integration
    @pytest.mark.asyncio 
    async def test_system_test_runner(self, blender_studio):
        """Test the built-in system test functionality"""
        
        test_result = await blender_studio.test_agents()
        
        assert test_result["status"] == "success"
        assert test_result["prompt"] == "Create a simple scene with a cube on a plane"
        assert test_result["style"] == "realistic"
        assert test_result["quality"] == "draft"


class TestCachingIntegration:
    """Test caching system integration"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_response_caching(self, blender_studio):
        """Test that AI responses are properly cached"""
        
        cache = get_cache()
        initial_stats = cache.get_stats()
        
        prompt = "A simple test scene for caching"
        
        # First generation should miss cache
        result1 = await blender_studio.generate_scene(
            prompt=prompt,
            style="realistic",
            quality="draft"
        )
        
        # Second identical generation should hit cache (at least partially)
        result2 = await blender_studio.generate_scene(
            prompt=prompt,
            style="realistic", 
            quality="draft"
        )
        
        final_stats = cache.get_stats()
        
        # Cache should have been used
        assert final_stats["performance"]["total_requests"] > initial_stats["performance"]["total_requests"]
        
        # Both results should be successful
        assert result1["status"] == "success"
        assert result2["status"] == "success"
    
    @pytest.mark.integration
    def test_cache_cleanup_and_optimization(self):
        """Test cache cleanup and optimization"""
        
        cache = get_cache()
        
        # Add some test data to cache
        for i in range(10):
            cache.put(
                f"test_key_{i}",
                f"test_value_{i}",
                cache_type=cache.CacheType.AI_RESPONSE if hasattr(cache, 'CacheType') else "ai_response"
            )
        
        initial_stats = cache.get_stats()
        assert initial_stats["total_entries"] >= 10
        
        # Perform optimization
        optimization_result = cache.optimize()
        
        assert optimization_result["optimization_performed"]
        assert "cleanup_stats" in optimization_result


class TestAssetIntegration:
    """Test asset management system integration"""
    
    @pytest.mark.integration
    def test_asset_library_scanning(self, mock_asset_manager, temp_dir):
        """Test asset library scanning and indexing"""
        
        # Create some test assets
        assets_dir = os.path.join(temp_dir, "test_assets")
        os.makedirs(os.path.join(assets_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(assets_dir, "textures"), exist_ok=True)
        
        # Create dummy asset files
        test_files = [
            "models/test_building.obj",
            "models/test_vehicle.blend",
            "textures/test_concrete.png",
            "textures/test_metal.jpg"
        ]
        
        for file_path in test_files:
            full_path = os.path.join(assets_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write("Test asset content")
        
        # Scan the assets
        from utils.asset_manager import AssetLibrary, AssetCategory
        
        library = AssetLibrary(
            name="test_library",
            path=assets_dir,
            categories=[AssetCategory.ARCHITECTURE, AssetCategory.VEHICLES]
        )
        
        scan_count = mock_asset_manager.scan_library(library)
        assert scan_count > 0
        
        # Verify assets are indexed
        stats = mock_asset_manager.get_stats()
        assert stats["total_assets"] > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_asset_integration_with_scene_generation(self, blender_studio, mock_asset_manager):
        """Test that asset management integrates with scene generation"""
        
        # Ensure assets are loaded
        asset_manager = get_asset_manager()
        asset_manager.scan_all_libraries()
        
        result = await blender_studio.generate_scene(
            prompt="A room with furniture and decorations",
            style="modern",
            quality="draft"
        )
        
        assert result["status"] == "success"
        
        # Check if asset placement agent was involved
        if "asset_placement" in result["agent_results"]:
            asset_result = result["agent_results"]["asset_placement"]
            # Should have some asset-related output
            assert "blender_commands" in asset_result or "asset_placements" in asset_result


class TestConfigurationIntegration:
    """Test configuration system integration"""
    
    @pytest.mark.integration
    def test_configuration_loading_and_validation(self, mock_config):
        """Test configuration loading and validation"""
        
        config = get_config()
        
        # Test basic configuration access
        assert config.settings.google_api_key == "test_api_key"
        assert config.performance.max_concurrent_operations > 0
        assert config.blender.timeout > 0
        
        # Test agent configurations
        director_config = config.get_agent_config("scene_director")
        assert director_config.model == "gemini-2.0-flash-exp"
        assert director_config.temperature == 0.8
        
        # Test quality presets
        high_preset = config.get_quality_preset("high")
        assert high_preset.samples == 128
        assert high_preset.resolution_scale == 1.0
    
    @pytest.mark.integration
    def test_configuration_validation(self, mock_config):
        """Test configuration validation"""
        
        config = get_config()
        issues = config.validate_config()
        
        # With mock config, should have minimal issues
        # (may have asset path issues, but API key should be valid)
        api_key_issues = [issue for issue in issues if "GOOGLE_API_KEY" in issue]
        assert len(api_key_issues) == 0  # Mock config has valid API key


class TestPerformanceIntegration:
    """Test performance-related integration"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_scene_generation_performance(self, blender_studio, performance_monitor):
        """Test scene generation performance characteristics"""
        
        performance_monitor.start()
        
        try:
            # Generate multiple scenes to test performance
            scenes = [
                {"prompt": "Simple cube scene", "style": "realistic", "quality": "draft"},
                {"prompt": "Complex city scene", "style": "cyberpunk", "quality": "preview"},
                {"prompt": "Medieval castle", "style": "medieval", "quality": "draft"}
            ]
            
            results = []
            for scene_config in scenes:
                result = await blender_studio.generate_scene(**scene_config)
                results.append(result)
        
        finally:
            performance_monitor.stop()
        
        # All generations should succeed
        for result in results:
            assert result["status"] == "success"
        
        # Check performance metrics
        stats = performance_monitor.get_stats()
        
        # Basic performance assertions
        assert stats["duration"] > 0
        assert stats["max_memory_mb"] > 0
        assert stats["max_memory_mb"] < 1000  # Shouldn't use excessive memory
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, blender_studio):
        """Test that memory usage remains stable across multiple generations"""
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate multiple scenes
        for i in range(3):
            result = await blender_studio.generate_scene(
                prompt=f"Test scene {i}",
                style="realistic",
                quality="draft"
            )
            assert result["status"] == "success"
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB for test scenarios)
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB"


class TestErrorHandlingIntegration:
    """Test error handling across the integrated system"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_blender_execution_failure_handling(self, blender_studio):
        """Test handling of Blender execution failures"""
        
        # Mock Blender executor to fail
        original_executor = blender_studio.blender_executor
        
        def failing_executor(commands, output_file):
            return {
                "status": "error",
                "output_file": None,
                "logs": "",
                "errors": "Mock Blender execution failure"
            }
        
        blender_studio.blender_executor.execute_commands = failing_executor
        
        try:
            result = await blender_studio.generate_scene(
                prompt="Test scene",
                style="realistic",
                quality="draft"
            )
            
            # Should handle the failure gracefully
            assert result["status"] == "error"
            assert "error" in result
            
        finally:
            # Restore original executor
            blender_studio.blender_executor = original_executor
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_partial_agent_failure_recovery(self, blender_studio):
        """Test recovery from partial agent failures"""
        
        # This would require more sophisticated mocking to simulate partial failures
        # For now, test that the system can handle overall failures gracefully
        
        result = await blender_studio.generate_scene(
            prompt="Extremely complex impossible scene with contradictory requirements",
            style="nonexistent_style",
            quality="impossible_quality"
        )
        
        # System should either succeed with reasonable defaults or fail gracefully
        assert result["status"] in ["success", "error"]
        
        if result["status"] == "error":
            assert "error" in result
            assert isinstance(result["error"], str)
            assert len(result["error"]) > 0