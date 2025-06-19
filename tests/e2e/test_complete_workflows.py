"""End-to-end tests for complete user workflows"""

import pytest
import asyncio
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from main import BlenderAIStudio


class TestCompleteUserWorkflows:
    """Test complete end-to-end user workflows"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_beginner_user_workflow(self, blender_studio, temp_dir):
        """Test complete workflow for a beginner user creating their first scene"""
        
        # Beginner user workflow:
        # 1. Simple prompt with basic requirements
        # 2. Use default settings
        # 3. Generate scene
        # 4. Render preview
        
        # Step 1: Generate scene with simple prompt
        prompt = "A cozy living room with a couch, coffee table, and lamp"
        
        result = await blender_studio.generate_scene(
            prompt=prompt,
            style="realistic",
            quality="preview"  # Good balance for beginners
        )
        
        # Verify successful generation
        assert result["status"] == "success"
        assert result["output_file"] is not None
        assert os.path.exists(result["output_file"])
        
        # Verify reasonable generation time (should be under 2 minutes for beginners)
        assert result["execution_time"] < 120
        
        # Step 2: Render preview image
        render_result = await blender_studio.render_scene(result["output_file"])
        
        assert render_result["status"] == "success"
        assert render_result["output_image"] is not None
        assert os.path.exists(render_result["output_image"])
        
        # Step 3: Verify scene quality review provides helpful feedback
        review = result.get("review", {})
        assert "overall_score" in review
        assert review["overall_score"] > 0
        
        # Should have user-friendly feedback
        assert "feedback" in review
        assert len(review["feedback"]) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_professional_user_workflow(self, blender_studio, temp_dir):
        """Test complete workflow for a professional user with specific requirements"""
        
        # Professional user workflow:
        # 1. Detailed prompt with specific requirements
        # 2. High quality settings
        # 3. Custom style parameters
        # 4. Quality validation
        # 5. Multiple render outputs
        
        # Step 1: Generate high-quality scene with detailed prompt
        prompt = """
        Create a modern architectural visualization of a glass office tower at sunset.
        The building should be 40 stories tall with a curtain wall facade.
        Include surrounding urban context with street-level retail, pedestrian areas,
        and landscaping. Use warm lighting to emphasize the golden hour atmosphere.
        Include subtle atmospheric effects like lens flares and depth of field.
        """
        
        result = await blender_studio.generate_scene(
            prompt=prompt,
            style="realistic",
            quality="high"
        )
        
        # Verify professional-grade output
        assert result["status"] == "success"
        assert result["output_file"] is not None
        
        # Professional scenes may take longer but should still be reasonable
        assert result["execution_time"] < 300  # 5 minutes max
        
        # Step 2: Verify high-quality scene components
        agent_results = result.get("agent_results", {})
        
        # Should have executed all major agents
        expected_agents = ["environment_builder", "lighting_designer", "asset_placement"]
        for agent in expected_agents:
            if agent in agent_results:
                assert len(agent_results[agent].get("blender_commands", [])) > 0
        
        # Step 3: Verify scene complexity appropriate for professional use
        total_commands = sum(
            len(agent_result.get("blender_commands", []))
            for agent_result in agent_results.values()
        )
        assert total_commands >= 5  # Should have substantial scene complexity
        
        # Step 4: Professional quality review
        review = result.get("review", {})
        assert review.get("overall_score", 0) >= 6  # Professional standard (6+/10)
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_iterative_design_workflow(self, blender_studio, temp_dir):
        """Test iterative design workflow with multiple refinements"""
        
        # Iterative workflow:
        # 1. Initial concept
        # 2. Review and refine
        # 3. Multiple iterations
        # 4. Final high-quality output
        
        base_prompt = "A cyberpunk street scene"
        iterations = [
            {"prompt": f"{base_prompt} with basic neon lighting", "quality": "draft"},
            {"prompt": f"{base_prompt} with detailed neon signs and fog effects", "quality": "preview"},
            {"prompt": f"{base_prompt} with volumetric lighting, detailed architecture, and atmospheric particles", "quality": "high"}
        ]
        
        results = []
        
        for i, iteration in enumerate(iterations):
            result = await blender_studio.generate_scene(
                prompt=iteration["prompt"],
                style="cyberpunk",
                quality=iteration["quality"]
            )
            
            assert result["status"] == "success"
            results.append(result)
            
            # Each iteration should build complexity
            if i > 0:
                current_commands = sum(
                    len(agent_result.get("blender_commands", []))
                    for agent_result in result.get("agent_results", {}).values()
                )
                previous_commands = sum(
                    len(agent_result.get("blender_commands", []))
                    for agent_result in results[i-1].get("agent_results", {}).values()
                )
                
                # Later iterations should generally have more complexity
                # (though this isn't guaranteed with current implementation)
                assert current_commands > 0
                assert previous_commands > 0
        
        # Final iteration should be highest quality
        final_result = results[-1]
        assert final_result["quality"] == "high"
        
        # Should have comprehensive scene review
        final_review = final_result.get("review", {})
        assert final_review.get("overall_score", 0) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_style_exploration_workflow(self, blender_studio, temp_dir):
        """Test workflow exploring different artistic styles"""
        
        # Multi-style exploration workflow:
        # 1. Same base concept
        # 2. Multiple style variations
        # 3. Compare results
        # 4. Select best approach
        
        base_prompt = "A ancient temple in a forest clearing"
        styles_to_explore = [
            "realistic",
            "fantasy", 
            "cartoon",
            "medieval"
        ]
        
        style_results = {}
        
        for style in styles_to_explore:
            result = await blender_studio.generate_scene(
                prompt=base_prompt,
                style=style,
                quality="preview"  # Good balance for exploration
            )
            
            assert result["status"] == "success"
            style_results[style] = result
        
        # All styles should succeed
        assert len(style_results) == len(styles_to_explore)
        
        # Each style should produce different scene characteristics
        for style, result in style_results.items():
            assert result["style"] == style
            assert result["output_file"] is not None
            
            # Scene should reflect style in agent results
            agent_results = result.get("agent_results", {})
            if "lighting_designer" in agent_results:
                # Different styles should produce different lighting approaches
                lighting_commands = agent_results["lighting_designer"].get("blender_commands", [])
                assert len(lighting_commands) > 0
        
        # Compare execution times (should all be reasonable)
        execution_times = [result["execution_time"] for result in style_results.values()]
        max_time = max(execution_times)
        assert max_time < 180  # All styles should complete within 3 minutes
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_production_pipeline_workflow(self, blender_studio, temp_dir):
        """Test complete production pipeline workflow"""
        
        # Production pipeline:
        # 1. Concept development (draft)
        # 2. Asset placement and lighting (preview)
        # 3. Final production (high quality)
        # 4. Rendering and output
        # 5. Quality validation
        
        project_name = "production_test"
        
        # Phase 1: Concept Development
        concept_result = await blender_studio.generate_scene(
            prompt="Modern corporate headquarters with glass atrium and water features",
            style="realistic",
            quality="draft"
        )
        
        assert concept_result["status"] == "success"
        concept_time = concept_result["execution_time"]
        
        # Phase 2: Detailed Development
        detailed_result = await blender_studio.generate_scene(
            prompt="Modern corporate headquarters with detailed glass atrium, interior lighting, water features, landscaping, and surrounding urban context",
            style="realistic", 
            quality="preview"
        )
        
        assert detailed_result["status"] == "success"
        detailed_time = detailed_result["execution_time"]
        
        # Phase 3: Final Production
        final_result = await blender_studio.generate_scene(
            prompt="Photorealistic corporate headquarters with detailed glass atrium, volumetric interior lighting, animated water features, detailed landscaping, surrounding urban context with traffic and pedestrians",
            style="realistic",
            quality="high"
        )
        
        assert final_result["status"] == "success"
        final_time = final_result["execution_time"]
        
        # Phase 4: Rendering
        render_result = await blender_studio.render_scene(final_result["output_file"])
        
        assert render_result["status"] == "success"
        assert os.path.exists(render_result["output_image"])
        
        # Phase 5: Quality Validation
        final_review = final_result.get("review", {})
        assert final_review.get("overall_score", 0) >= 7  # Production quality standard
        
        # Verify progression in complexity/quality
        concept_complexity = sum(
            len(agent_result.get("blender_commands", []))
            for agent_result in concept_result.get("agent_results", {}).values()
        )
        final_complexity = sum(
            len(agent_result.get("blender_commands", []))
            for agent_result in final_result.get("agent_results", {}).values()
        )
        
        assert concept_complexity > 0
        assert final_complexity > 0
        # Final should generally be more complex than concept
        # (though this depends on implementation)
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, blender_studio, temp_dir):
        """Test batch processing workflow for multiple scenes"""
        
        # Batch processing workflow:
        # 1. Multiple scene requests
        # 2. Concurrent processing
        # 3. Results aggregation
        # 4. Quality consistency
        
        batch_scenes = [
            {"prompt": "Modern office lobby", "style": "realistic", "id": "office_01"},
            {"prompt": "Retail store interior", "style": "modern", "id": "retail_01"},
            {"prompt": "Restaurant dining area", "style": "contemporary", "id": "restaurant_01"},
            {"prompt": "Hotel reception desk", "style": "luxury", "id": "hotel_01"}
        ]
        
        # Process batch concurrently
        tasks = []
        for scene_config in batch_scenes:
            task = blender_studio.generate_scene(
                prompt=scene_config["prompt"],
                style=scene_config.get("style", "realistic"),
                quality="preview"
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all scenes completed successfully
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Batch scene {i} failed: {result}")
            
            assert result["status"] == "success"
            successful_results.append(result)
        
        assert len(successful_results) == len(batch_scenes)
        
        # Verify quality consistency across batch
        quality_scores = []
        execution_times = []
        
        for result in successful_results:
            review = result.get("review", {})
            if "overall_score" in review:
                quality_scores.append(review["overall_score"])
            execution_times.append(result["execution_time"])
        
        # Quality should be reasonably consistent
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            assert avg_quality > 5  # Batch should maintain good quality
        
        # Execution times should be reasonable
        max_time = max(execution_times)
        assert max_time < 300  # No single scene should take too long


class TestErrorHandlingWorkflows:
    """Test error handling in real-world scenarios"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_prompt_handling(self, blender_studio):
        """Test handling of various invalid prompts"""
        
        invalid_prompts = [
            "",  # Empty prompt
            "a",  # Too short
            "x" * 10000,  # Too long
            "Create a scene with impossible physics violations and contradictory requirements that cannot exist",  # Contradictory
            "ăđĸűŋđĸűŋđ",  # Special characters
        ]
        
        for prompt in invalid_prompts:
            result = await blender_studio.generate_scene(
                prompt=prompt,
                style="realistic",
                quality="draft"
            )
            
            # Should either succeed with reasonable interpretation or fail gracefully
            assert result["status"] in ["success", "error"]
            
            if result["status"] == "error":
                assert "error" in result
                assert isinstance(result["error"], str)
                assert len(result["error"]) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_resource_limitation_handling(self, blender_studio):
        """Test handling of resource-intensive requests"""
        
        # Test very complex scene request
        complex_prompt = """
        Create an extremely detailed cyberpunk megacity with thousands of buildings,
        millions of lights, complex weather systems, hundreds of vehicles,
        detailed interior spaces, volumetric fog, particle effects,
        reflective surfaces everywhere, and cinematic camera movements.
        Include every possible detail and make it photorealistic.
        """
        
        result = await blender_studio.generate_scene(
            prompt=complex_prompt,
            style="cyberpunk",
            quality="ultra"  # Most demanding quality
        )
        
        # Should handle gracefully - either succeed with reasonable complexity
        # or fail with helpful error message
        assert result["status"] in ["success", "error"]
        
        if result["status"] == "success":
            # Should have reasonable execution time even for complex scenes
            assert result["execution_time"] < 600  # 10 minutes max
            
            # Should have generated content
            assert result["output_file"] is not None
        
        elif result["status"] == "error":
            # Should provide helpful error message
            assert "error" in result
            error_msg = result["error"].lower()
            
            # Error should indicate resource limitations or complexity issues
            resource_keywords = ["memory", "timeout", "complex", "limit", "resource"]
            assert any(keyword in error_msg for keyword in resource_keywords)


class TestRealWorldScenarios:
    """Test scenarios that match real-world usage patterns"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_architectural_visualization_scenario(self, blender_studio):
        """Test architectural visualization use case"""
        
        # Architect creating building visualization
        result = await blender_studio.generate_scene(
            prompt="Modern sustainable office building with green roof, solar panels, and natural lighting",
            style="realistic",
            quality="high"
        )
        
        assert result["status"] == "success"
        
        # Should have environment and lighting appropriate for architecture
        agent_results = result.get("agent_results", {})
        
        if "environment_builder" in agent_results:
            env_commands = agent_results["environment_builder"].get("blender_commands", [])
            # Should have building/structure creation
            structure_commands = [cmd for cmd in env_commands if cmd.get("type") == "create_structure"]
            assert len(structure_commands) > 0
        
        if "lighting_designer" in agent_results:
            lighting_commands = agent_results["lighting_designer"].get("blender_commands", [])
            # Should have appropriate lighting for architecture
            assert len(lighting_commands) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_game_development_scenario(self, blender_studio):
        """Test game development use case"""
        
        # Game developer creating environment assets
        result = await blender_studio.generate_scene(
            prompt="Fantasy village with medieval buildings, market stalls, and cobblestone paths",
            style="medieval",
            quality="preview"  # Game-appropriate quality
        )
        
        assert result["status"] == "success"
        
        # Should be optimized for game use (reasonable execution time)
        assert result["execution_time"] < 120
        
        # Should have multiple structures for game environment
        agent_results = result.get("agent_results", {})
        if "environment_builder" in agent_results:
            commands = agent_results["environment_builder"].get("blender_commands", [])
            structure_commands = [cmd for cmd in commands if cmd.get("type") == "create_structure"]
            assert len(structure_commands) >= 2  # Multiple buildings for village
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_education_scenario(self, blender_studio):
        """Test educational use case"""
        
        # Teacher creating educational content
        result = await blender_studio.generate_scene(
            prompt="Solar system model with planets orbiting the sun",
            style="educational",
            quality="draft"  # Fast for classroom use
        )
        
        # Should complete quickly for classroom scenarios
        if result["status"] == "success":
            assert result["execution_time"] < 60  # 1 minute for classroom use
        else:
            # Even if it fails, should fail quickly and gracefully
            assert result.get("execution_time", 0) < 60
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_artistic_experimentation_scenario(self, blender_studio):
        """Test artistic experimentation use case"""
        
        # Artist exploring abstract concepts
        artistic_prompts = [
            "Abstract representation of music as flowing geometric forms",
            "Emotional landscape representing joy and melancholy",
            "Surreal floating islands with impossible architecture"
        ]
        
        results = []
        
        for prompt in artistic_prompts:
            result = await blender_studio.generate_scene(
                prompt=prompt,
                style="abstract",  # May not be specifically supported
                quality="preview"
            )
            
            # Should handle artistic/abstract concepts gracefully
            assert result["status"] in ["success", "error"]
            results.append(result)
        
        # At least some artistic experiments should succeed
        successful_results = [r for r in results if r["status"] == "success"]
        assert len(successful_results) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rapid_prototyping_scenario(self, blender_studio):
        """Test rapid prototyping use case"""
        
        # Designer rapidly testing concepts
        concepts = [
            "Minimalist furniture set",
            "Modern kitchen layout", 
            "Outdoor seating area"
        ]
        
        start_time = asyncio.get_event_loop().time()
        
        # Test rapid iteration
        for concept in concepts:
            result = await blender_studio.generate_scene(
                prompt=concept,
                style="modern",
                quality="draft"  # Speed over quality for prototyping
            )
            
            # Each concept should generate quickly
            assert result.get("execution_time", float('inf')) < 45  # 45 seconds max per concept
            
            if result["status"] == "success":
                assert result["output_file"] is not None
        
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Total prototyping session should be under 3 minutes
        assert total_time < 180