"""Main Blender AI Agent Studio orchestrator"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from agents import (
    SceneDirectorAgent,
    EnvironmentBuilderAgent, 
    LightingDesignerAgent,
    AssetPlacementAgent,
    EffectsCoordinatorAgent
)
from blender_integration import BlenderExecutor
from utils.logging_config import setup_logging, get_logger


class BlenderAIStudio:
    """Main orchestrator for the Blender AI Agent Studio"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Setup logging
        setup_logging()
        self.logger = get_logger("blender_ai_studio")
        
        # Initialize agents
        self.scene_director = SceneDirectorAgent()
        self.environment_builder = EnvironmentBuilderAgent()
        self.lighting_designer = LightingDesignerAgent()
        self.asset_placement = AssetPlacementAgent()
        self.effects_coordinator = EffectsCoordinatorAgent()
        
        # Initialize Blender executor
        self.blender_executor = BlenderExecutor()
        
        # Agent registry
        self.agents = {
            "scene_director": self.scene_director,
            "environment_builder": self.environment_builder,
            "lighting_designer": self.lighting_designer,
            "asset_placement": self.asset_placement,
            "effects_coordinator": self.effects_coordinator
        }
        
        self.logger.info("Blender AI Studio initialized")
        
    async def generate_scene(
        self,
        prompt: str,
        style: Optional[str] = None,
        quality: str = "high",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a complete 3D scene from natural language prompt"""
        
        start_time = time.time()
        scene_id = f"scene_{int(start_time)}"
        
        if output_file is None:
            output_file = f"./output/{scene_id}.blend"
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        self.logger.info(f"Starting scene generation: {prompt[:100]}...")
        
        try:
            # Step 1: Scene analysis and planning
            self.logger.info("Step 1: Analyzing prompt and creating execution plan")
            direction_result = await self.scene_director.execute({
                "prompt": prompt,
                "style": style,
                "quality": quality,
                "scene_id": scene_id
            })
            
            coordination_data = direction_result["coordination_data"]
            execution_order = coordination_data["execution_order"]
            
            # Step 2: Execute agents in order
            self.logger.info("Step 2: Executing specialist agents")
            all_commands = []
            agent_results = {}
            
            for phase in execution_order:
                # Execute agents in this phase in parallel
                phase_tasks = []
                
                for agent_name in phase:
                    if agent_name in self.agents:
                        # Find the task for this agent
                        agent_task = None
                        for task in coordination_data["tasks"]:
                            if task["agent_name"] == agent_name:
                                agent_task = task
                                break
                                
                        if agent_task:
                            # Prepare task data with shared state
                            task_data = {
                                "parameters": agent_task["parameters"],
                                "shared_state": {
                                    "metadata": coordination_data["metadata"],
                                    "previous_results": agent_results
                                }
                            }
                            
                            phase_tasks.append(
                                self._execute_agent_with_retry(agent_name, task_data)
                            )
                
                # Wait for all agents in this phase to complete
                if phase_tasks:
                    phase_results = await asyncio.gather(*phase_tasks, return_exceptions=True)
                    
                    for i, result in enumerate(phase_results):
                        agent_name = phase[i]
                        if isinstance(result, Exception):
                            self.logger.error(f"Agent {agent_name} failed: {result}")
                            raise result
                        else:
                            agent_results[agent_name] = result
                            # Collect Blender commands
                            if "blender_commands" in result:
                                all_commands.extend(result["blender_commands"])
                                
            # Step 3: Execute Blender commands
            self.logger.info("Step 3: Executing Blender scene generation")
            blender_result = self.blender_executor.execute_commands(all_commands, output_file)
            
            # Step 4: Quality review (optional)
            self.logger.info("Step 4: Scene quality review")
            review_data = {
                "requirements": coordination_data["requirements"],
                "completed_tasks": list(agent_results.keys()),
                "blender_result": blender_result
            }
            review_result = await self.scene_director.review_scene(review_data)
            
            # Compile final result
            total_time = time.time() - start_time
            
            result = {
                "status": "success" if blender_result["status"] == "success" else "error",
                "scene_id": scene_id,
                "output_file": blender_result["output_file"],
                "execution_time": total_time,
                "prompt": prompt,
                "style": coordination_data["requirements"]["style"],
                "quality": quality,
                "agent_results": agent_results,
                "blender_result": blender_result,
                "review": review_result,
                "statistics": {
                    "agents_executed": len(agent_results),
                    "commands_executed": len(all_commands),
                    "total_time": total_time
                }
            }
            
            self.logger.info(f"Scene generation completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Scene generation failed: {e}")
            return {
                "status": "error",
                "scene_id": scene_id,
                "output_file": None,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    async def _execute_agent_with_retry(
        self, 
        agent_name: str, 
        task_data: Dict[str, Any],
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """Execute agent with retry logic"""
        
        agent = self.agents[agent_name]
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await agent.execute(task_data)
                self.logger.info(f"Agent {agent_name} completed successfully")
                return result
                
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    self.logger.warning(f"Agent {agent_name} failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(1)  # Brief delay before retry
                else:
                    self.logger.error(f"Agent {agent_name} failed after {max_retries + 1} attempts: {e}")
                    
        raise last_error
        
    async def render_scene(
        self, 
        scene_file: str, 
        output_image: Optional[str] = None
    ) -> Dict[str, Any]:
        """Render a scene file to an image"""
        
        if output_image is None:
            base_name = os.path.splitext(scene_file)[0]
            output_image = f"{base_name}.png"
            
        self.logger.info(f"Rendering scene: {scene_file}")
        
        result = self.blender_executor.render_scene(scene_file, output_image)
        
        if result["status"] == "success":
            self.logger.info(f"Render completed: {output_image}")
        else:
            self.logger.error(f"Render failed: {result['errors']}")
            
        return result
        
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        status = {}
        
        for name, agent in self.agents.items():
            status[name] = agent.get_state()
            
        return status
        
    async def test_agents(self) -> Dict[str, Any]:
        """Test all agents with a simple prompt"""
        test_prompt = "Create a simple scene with a cube on a plane"
        
        self.logger.info("Running agent test...")
        
        result = await self.generate_scene(
            prompt=test_prompt,
            style="realistic",
            quality="draft"
        )
        
        return result


# CLI interface
async def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Blender AI Agent Studio")
    parser.add_argument("prompt", help="Scene description prompt")
    parser.add_argument("--style", help="Visual style (cyberpunk, medieval, etc.)")
    parser.add_argument("--quality", default="high", choices=["draft", "preview", "high", "ultra"])
    parser.add_argument("--output", help="Output .blend file path")
    parser.add_argument("--render", action="store_true", help="Also render to image")
    parser.add_argument("--test", action="store_true", help="Run system test")
    
    args = parser.parse_args()
    
    studio = BlenderAIStudio()
    
    if args.test:
        result = await studio.test_agents()
    else:
        result = await studio.generate_scene(
            prompt=args.prompt,
            style=args.style,
            quality=args.quality,
            output_file=args.output
        )
        
        if args.render and result["status"] == "success":
            render_result = await studio.render_scene(result["output_file"])
            result["render_result"] = render_result
    
    print(f"Result: {result['status']}")
    if result["status"] == "success":
        print(f"Output: {result['output_file']}")
        print(f"Time: {result['execution_time']:.2f}s")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())