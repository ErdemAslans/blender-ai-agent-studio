"""Scene Director Agent - Main orchestrator for scene generation"""

import json
from typing import Any, Dict, List, Optional, Tuple

from google.genai.adk import tool
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentConfig


class SceneRequirements(BaseModel):
    """Parsed scene requirements from natural language"""
    environment_type: str = Field(..., description="Type of environment (city, forest, indoor, etc.)")
    style: str = Field(..., description="Visual style (cyberpunk, medieval, realistic, etc.)")
    time_of_day: str = Field("day", description="Time setting (day, night, dawn, dusk)")
    weather: Optional[str] = Field(None, description="Weather conditions")
    mood: str = Field("neutral", description="Overall mood/atmosphere")
    key_elements: List[str] = Field(default_factory=list, description="Key objects/elements to include")
    lighting_notes: str = Field("", description="Special lighting requirements")
    camera_hints: Optional[Dict[str, Any]] = Field(None, description="Camera positioning hints")
    complexity: str = Field("medium", description="Scene complexity level")


class AgentTask(BaseModel):
    """Task assignment for a specific agent"""
    agent_name: str
    task_description: str
    parameters: Dict[str, Any]
    priority: int = Field(1, ge=1, le=5)
    dependencies: List[str] = Field(default_factory=list)


class SceneDirectorAgent(BaseAgent):
    """
    The Scene Director Agent acts as the primary orchestrator that understands 
    artistic intent and delegates specific tasks to specialist agents.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="scene_director",
                model="gemini-2.0-flash-exp",
                temperature=0.8,
                max_tokens=4096
            )
        super().__init__(config)
        
        # Define prompts for scene analysis
        self.analysis_prompt_template = """
        Analyze the following scene description and extract structured requirements:
        
        Description: {description}
        
        Please identify:
        1. Environment type (city, forest, indoor, space, etc.)
        2. Visual style (cyberpunk, medieval, realistic, fantasy, etc.)
        3. Time of day (day, night, dawn, dusk)
        4. Weather conditions (if mentioned)
        5. Overall mood/atmosphere
        6. Key elements that must be included
        7. Special lighting requirements
        8. Camera positioning hints (if any)
        9. Scene complexity (simple, medium, complex)
        
        Respond in JSON format matching the SceneRequirements schema.
        """
        
        self.task_delegation_prompt_template = """
        Based on these scene requirements, create specific tasks for each specialist agent:
        
        Scene Requirements:
        {requirements}
        
        Available Agents:
        1. Environment Builder - Creates structures and terrain
        2. Lighting Designer - Handles all lighting and atmosphere
        3. Asset Placement - Positions objects and props
        4. Effects Coordinator - Manages particles and weather
        
        For each agent, specify:
        - Detailed task description
        - Specific parameters they should use
        - Priority (1-5, where 5 is highest)
        - Dependencies on other agents
        
        Respond with a JSON list of AgentTask objects.
        """
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process scene generation request"""
        prompt = input_data.get("prompt", "")
        style_override = input_data.get("style")
        quality = input_data.get("quality", "high")
        
        self.logger.info(f"Processing scene prompt: {prompt[:100]}...")
        
        # Step 1: Analyze the prompt and extract requirements
        requirements = await self._analyze_prompt(prompt, style_override)
        self.update_metadata("scene_requirements", requirements.model_dump())
        
        # Step 2: Create task delegation plan
        agent_tasks = await self._create_task_plan(requirements, quality)
        self.update_metadata("task_plan", [task.model_dump() for task in agent_tasks])
        
        # Step 3: Organize execution order based on dependencies
        execution_order = self._organize_execution_order(agent_tasks)
        
        # Step 4: Prepare coordination data
        coordination_data = {
            "scene_id": f"scene_{input_data.get('scene_id', 'default')}",
            "requirements": requirements.model_dump(),
            "tasks": [task.model_dump() for task in agent_tasks],
            "execution_order": execution_order,
            "quality_preset": quality,
            "metadata": {
                "original_prompt": prompt,
                "style": requirements.style,
                "complexity": requirements.complexity
            }
        }
        
        return {
            "status": "success",
            "coordination_data": coordination_data,
            "message": f"Scene analysis complete. Prepared {len(agent_tasks)} tasks for execution."
        }
    
    @tool
    async def analyze_scene_prompt(self, prompt: str, style: str = None, quality: str = "high") -> str:
        """Analyze scene prompt and create task plan - ADK tool format"""
        try:
            input_data = {
                "prompt": prompt,
                "style": style,
                "quality": quality,
                "scene_id": f"scene_{hash(prompt) % 10000}"
            }
            
            result = await self.process(input_data)
            return json.dumps(result)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "message": "Failed to analyze scene prompt"
            })
    
    @tool
    async def review_generated_scene(self, scene_data: str) -> str:
        """Review generated scene - ADK tool format"""
        try:
            scene_dict = json.loads(scene_data)
            result = await self.review_scene(scene_dict)
            return json.dumps(result)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "message": "Failed to review scene"
            })
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using ADK's built-in capabilities"""
        # This would use ADK's built-in LLM integration
        # For now, we'll use a placeholder
        return await self.generate_content(prompt)
        
    async def _analyze_prompt(self, prompt: str, style_override: Optional[str] = None) -> SceneRequirements:
        """Analyze natural language prompt and extract structured requirements"""
        analysis_prompt = self.analysis_prompt_template.format(description=prompt)
        
        # Use ADK's built-in response generation
        response = await self._generate_response(analysis_prompt)
        
        try:
            # Parse JSON response
            requirements_data = json.loads(response)
            requirements = SceneRequirements(**requirements_data)
            
            # Apply style override if provided
            if style_override:
                requirements.style = style_override
                
            return requirements
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse requirements: {e}")
            # Fallback to basic requirements
            return SceneRequirements(
                environment_type="generic",
                style=style_override or "realistic",
                key_elements=self._extract_key_elements(prompt)
            )
            
    async def _create_task_plan(self, requirements: SceneRequirements, quality: str) -> List[AgentTask]:
        """Create detailed task plan for each agent"""
        delegation_prompt = self.task_delegation_prompt_template.format(
            requirements=requirements.model_dump_json(indent=2)
        )
        
        response = await self._generate_response(delegation_prompt)
        
        try:
            tasks_data = json.loads(response)
            tasks = [AgentTask(**task_data) for task_data in tasks_data]
            return tasks
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse task plan: {e}")
            # Fallback to default task plan
            return self._create_default_task_plan(requirements, quality)
            
    def _create_default_task_plan(self, requirements: SceneRequirements, quality: str) -> List[AgentTask]:
        """Create a default task plan as fallback"""
        tasks = []
        
        # Environment Builder task
        tasks.append(AgentTask(
            agent_name="environment_builder",
            task_description=f"Create {requirements.environment_type} environment with {requirements.style} style",
            parameters={
                "environment_type": requirements.environment_type,
                "style": requirements.style,
                "complexity": requirements.complexity,
                "quality": quality
            },
            priority=5,
            dependencies=[]
        ))
        
        # Lighting Designer task
        tasks.append(AgentTask(
            agent_name="lighting_designer",
            task_description=f"Set up {requirements.time_of_day} lighting with {requirements.mood} mood",
            parameters={
                "time_of_day": requirements.time_of_day,
                "mood": requirements.mood,
                "style": requirements.style,
                "special_requirements": requirements.lighting_notes
            },
            priority=4,
            dependencies=["environment_builder"]
        ))
        
        # Asset Placement task
        if requirements.key_elements:
            tasks.append(AgentTask(
                agent_name="asset_placement",
                task_description=f"Place key elements: {', '.join(requirements.key_elements)}",
                parameters={
                    "elements": requirements.key_elements,
                    "environment_type": requirements.environment_type,
                    "style": requirements.style
                },
                priority=3,
                dependencies=["environment_builder"]
            ))
            
        # Effects Coordinator task
        if requirements.weather or requirements.style in ["cyberpunk", "post-apocalyptic"]:
            tasks.append(AgentTask(
                agent_name="effects_coordinator",
                task_description=f"Add weather/atmospheric effects",
                parameters={
                    "weather": requirements.weather,
                    "atmosphere": requirements.mood,
                    "style": requirements.style
                },
                priority=2,
                dependencies=["environment_builder", "lighting_designer"]
            ))
            
        return tasks
        
    def _organize_execution_order(self, tasks: List[AgentTask]) -> List[List[str]]:
        """Organize tasks into execution phases based on dependencies"""
        # Create dependency graph
        task_map = {task.agent_name: task for task in tasks}
        completed = set()
        execution_order = []
        
        while len(completed) < len(tasks):
            current_phase = []
            
            for task in tasks:
                if task.agent_name in completed:
                    continue
                    
                # Check if all dependencies are completed
                if all(dep in completed for dep in task.dependencies):
                    current_phase.append(task.agent_name)
                    
            if not current_phase:
                # Break circular dependencies
                remaining = [t.agent_name for t in tasks if t.agent_name not in completed]
                current_phase = remaining[:1]
                
            execution_order.append(current_phase)
            completed.update(current_phase)
            
        return execution_order
        
    def _extract_key_elements(self, prompt: str) -> List[str]:
        """Extract potential key elements from prompt"""
        # Simple keyword extraction as fallback
        keywords = ["building", "tree", "car", "person", "light", "sign", "road", 
                   "sky", "ground", "wall", "door", "window", "bridge", "vehicle"]
        
        elements = []
        prompt_lower = prompt.lower()
        
        for keyword in keywords:
            if keyword in prompt_lower:
                elements.append(keyword)
                
        return elements[:5]  # Limit to 5 elements
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return "prompt" in input_data and isinstance(input_data["prompt"], str)
    
    @tool
    def validate_scene_input(self, input_data: str) -> str:
        """Validate scene input data - ADK tool format"""
        try:
            import json
            data = json.loads(input_data)
            is_valid = self.validate_input(data)
            return json.dumps({
                "valid": is_valid,
                "message": "Input is valid" if is_valid else "Input requires 'prompt' field"
            })
        except Exception as e:
            return json.dumps({
                "valid": False,
                "error": str(e),
                "message": "Invalid JSON input"
            })
        
    async def review_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Review and provide feedback on generated scene"""
        review_prompt = f"""
        Review the following generated scene data and provide feedback:
        
        Original Requirements: {scene_data.get('requirements', {})}
        Completed Tasks: {scene_data.get('completed_tasks', [])}
        
        Evaluate:
        1. Does the scene match the original requirements?
        2. Are all key elements present?
        3. Is the mood/atmosphere appropriate?
        4. Any missing or incorrect elements?
        5. Overall quality assessment (1-10)
        
        Provide constructive feedback and suggestions for improvement.
        """
        
        feedback = await self._generate_response(review_prompt)
        
        return {
            "status": "reviewed",
            "feedback": feedback,
            "approval": True  # Can be made more sophisticated
        }