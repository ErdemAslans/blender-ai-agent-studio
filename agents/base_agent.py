"""Base Agent Class for Blender AI Agent Studio - ADK Compatible"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from google.genai.adk import Agent, tool
from pydantic import BaseModel, Field

from utils.logging_config import get_logger


class AgentConfig(BaseModel):
    """Configuration for an agent"""
    name: str
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 3
    timeout: int = 60


class AgentState(BaseModel):
    """State information for an agent"""
    agent_id: str
    current_task: Optional[str] = None
    completed_tasks: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(Agent):
    """Base class for all agents in the Blender AI Agent Studio"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(name=config.name)
        self.config = config
        self.logger = get_logger(f"agent.{config.name}")
        self.state = AgentState(agent_id=config.name)
        
    @tool
    async def process_scene_task(self, task_data: str) -> str:
        """Process scene generation task - ADK tool format"""
        try:
            import json
            task = json.loads(task_data)
            
            # Set current task
            self.state.current_task = task.get("description", "Unknown task")
            self.logger.info(f"Starting task: {self.state.current_task}")
            
            # Process task
            result = await self.process(task)
            
            # Mark task as completed
            self.state.completed_tasks.append(self.state.current_task)
            self.state.current_task = None
            
            self.logger.info(f"Completed task successfully")
            return json.dumps(result)
            
        except Exception as e:
            error_msg = f"Error executing task: {str(e)}"
            self.logger.error(error_msg)
            self.state.errors.append(error_msg)
            return json.dumps({"status": "error", "error": error_msg})
    
    @tool
    def get_agent_state(self) -> str:
        """Get current agent state - ADK tool format"""
        return json.dumps(self.state.model_dump())
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results - to be implemented by subclasses"""
        pass
    
    @tool
    def reset_agent_state(self) -> str:
        """Reset agent state - ADK tool format"""
        self.state = AgentState(agent_id=self.config.name)
        self.logger.info("Agent state reset")
        return json.dumps({"status": "reset", "message": "Agent state has been reset"})
        
    @tool
    def update_agent_metadata(self, key: str, value: str) -> str:
        """Update agent metadata - ADK tool format"""
        import json
        try:
            parsed_value = json.loads(value)
        except:
            parsed_value = value
        
        self.state.metadata[key] = parsed_value
        return json.dumps({"status": "updated", "key": key, "value": parsed_value})
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return self.state.model_dump()
        
    def update_metadata(self, key: str, value: Any):
        """Update agent metadata"""
        self.state.metadata[key] = value