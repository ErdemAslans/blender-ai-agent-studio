"""Pytest configuration and fixtures for Blender AI Agent Studio tests"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import BlenderAIStudio
from agents.base_agent import BaseAgent, AgentConfig
from agents.scene_director import SceneDirectorAgent
from agents.environment_builder import EnvironmentBuilderAgent
from agents.lighting_designer import LightingDesignerAgent
from agents.asset_placement import AssetPlacementAgent
from agents.effects_coordinator import EffectsCoordinatorAgent
from blender_integration.blender_executor import BlenderExecutor
from utils.config import ConfigManager
from utils.asset_manager import AssetManager
from utils.caching import SmartCache
from utils.prompt_engineering import PromptEngineer


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_dir) -> ConfigManager:
    """Create a mock configuration for testing"""
    config_path = os.path.join(temp_dir, "test_config.yaml")
    
    config_content = """
agents:
  scene_director:
    model: "gemini-2.0-flash-exp"
    temperature: 0.8
    max_retries: 3
    timeout: 60
    
  environment_builder:
    model: "gemini-2.0-flash-exp"
    temperature: 0.6
    
  lighting_designer:
    model: "gemini-2.0-flash-exp"
    temperature: 0.7
    
  asset_placement:
    model: "gemini-2.0-flash-exp"
    temperature: 0.6
    
  effects_coordinator:
    model: "gemini-2.0-flash-exp"
    temperature: 0.7

scene_generation:
  default_resolution: [1920, 1080]
  default_samples: 128
  max_scene_complexity: "high"
  auto_optimize: true

quality_presets:
  draft:
    samples: 32
    resolution_scale: 0.5
    simplify_subdivision: 2
  high:
    samples: 128
    resolution_scale: 1.0
    simplify_subdivision: 0

asset_libraries:
  - name: "test"
    path: "{temp_dir}/assets"
    categories: ["props", "architecture"]
""".format(temp_dir=temp_dir)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Mock environment variables
    with patch.dict(os.environ, {
        'GOOGLE_API_KEY': 'test_api_key',
        'OUTPUT_DIRECTORY': os.path.join(temp_dir, 'output'),
        'LOG_LEVEL': 'DEBUG',
        'CACHE_SIZE_MB': '100',
        'DEVELOPMENT_MODE': 'true'
    }):
        config = ConfigManager(config_path)
        yield config


@pytest.fixture
def mock_api_response():
    """Mock AI API response"""
    return {
        "scene_analysis": """
        {
            "scene_type": "outdoor",
            "complexity": "moderate",
            "elements": ["buildings", "street", "vehicles"],
            "environment": {
                "terrain": "urban",
                "weather": "clear",
                "time_of_day": "day"
            },
            "mood": "modern urban",
            "style_requirements": {
                "architecture": "contemporary",
                "materials": "concrete and glass"
            },
            "agent_tasks": {
                "environment_builder": {"create_urban_layout": true},
                "lighting_designer": {"setup_daylight": true},
                "asset_placement": {"place_urban_assets": true}
            },
            "execution_order": [
                ["environment_builder"],
                ["lighting_designer", "asset_placement"],
                ["effects_coordinator"]
            ]
        }
        """,
        "environment_commands": [
            {
                "type": "create_terrain",
                "params": {
                    "size": [100, 100],
                    "subdivision": 32,
                    "material": "concrete"
                }
            },
            {
                "type": "create_structure",
                "params": {
                    "structure_type": "building",
                    "position": [0, 0, 0],
                    "scale": [10, 10, 30],
                    "material": "glass_concrete"
                }
            }
        ],
        "lighting_commands": [
            {
                "type": "create_light",
                "params": {
                    "light_type": "sun",
                    "position": [50, 50, 100],
                    "energy": 1000,
                    "color": [1, 1, 1]
                }
            }
        ]
    }


@pytest.fixture
def mock_blender_executor(temp_dir):
    """Create a mock Blender executor that doesn't require actual Blender"""
    executor = Mock(spec=BlenderExecutor)
    
    def mock_execute_commands(commands, output_file):
        # Create a dummy output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("# Mock Blender file\n")
        
        return {
            "status": "success",
            "output_file": output_file,
            "logs": "Mock Blender execution completed",
            "errors": None
        }
    
    def mock_render_scene(scene_file, output_image):
        # Create a dummy image file
        os.makedirs(os.path.dirname(output_image), exist_ok=True)
        with open(output_image, 'w') as f:
            f.write("Mock image data")
        
        return {
            "status": "success",
            "output_image": output_image,
            "logs": "Mock render completed",
            "errors": None
        }
    
    executor.execute_commands = Mock(side_effect=mock_execute_commands)
    executor.render_scene = Mock(side_effect=mock_render_scene)
    
    return executor


@pytest.fixture
def mock_gemini_api():
    """Mock Gemini API responses"""
    mock_model = Mock()
    
    def mock_generate_content(prompt):
        mock_response = Mock()
        
        # Return different responses based on prompt content
        if "scene analysis" in prompt.lower() or "analyze this scene" in prompt.lower():
            mock_response.text = """
            {
                "scene_type": "outdoor",
                "complexity": "moderate",
                "elements": ["buildings", "street", "vehicles"],
                "environment": {
                    "terrain": "urban",
                    "weather": "clear",
                    "time_of_day": "day"
                },
                "mood": "modern urban",
                "style_requirements": {
                    "architecture": "contemporary",
                    "materials": "concrete and glass"
                },
                "agent_tasks": {
                    "environment_builder": {"create_urban_layout": true},
                    "lighting_designer": {"setup_daylight": true},
                    "asset_placement": {"place_urban_assets": true}
                },
                "execution_order": [
                    ["environment_builder"],
                    ["lighting_designer", "asset_placement"],
                    ["effects_coordinator"]
                ]
            }
            """
        elif "environment" in prompt.lower():
            mock_response.text = "Urban environment with modern buildings and concrete surfaces"
        elif "lighting" in prompt.lower():
            mock_response.text = "Daylight setup with sun positioned high, creating natural shadows"
        elif "asset" in prompt.lower():
            mock_response.text = "Place urban furniture, vehicles, and street elements strategically"
        elif "effects" in prompt.lower():
            mock_response.text = "Add subtle atmospheric effects and environmental details"
        else:
            mock_response.text = "Generic AI response for testing"
        
        return mock_response
    
    mock_model.generate_content = Mock(side_effect=mock_generate_content)
    
    with patch('google.generativeai.GenerativeModel', return_value=mock_model):
        with patch('google.generativeai.configure'):
            yield mock_model


@pytest.fixture
def test_agent_config() -> AgentConfig:
    """Create a test agent configuration"""
    return AgentConfig(
        name="test_agent",
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        max_tokens=2048,
        max_retries=2,
        timeout=30
    )


@pytest.fixture
def scene_director_agent(test_agent_config, mock_gemini_api) -> SceneDirectorAgent:
    """Create a Scene Director agent for testing"""
    return SceneDirectorAgent(test_agent_config)


@pytest.fixture
def environment_builder_agent(test_agent_config, mock_gemini_api) -> EnvironmentBuilderAgent:
    """Create an Environment Builder agent for testing"""
    return EnvironmentBuilderAgent(test_agent_config)


@pytest.fixture
def lighting_designer_agent(test_agent_config, mock_gemini_api) -> LightingDesignerAgent:
    """Create a Lighting Designer agent for testing"""
    return LightingDesignerAgent(test_agent_config)


@pytest.fixture
def asset_placement_agent(test_agent_config, mock_gemini_api) -> AssetPlacementAgent:
    """Create an Asset Placement agent for testing"""
    return AssetPlacementAgent(test_agent_config)


@pytest.fixture
def effects_coordinator_agent(test_agent_config, mock_gemini_api) -> EffectsCoordinatorAgent:
    """Create an Effects Coordinator agent for testing"""
    return EffectsCoordinatorAgent(test_agent_config)


@pytest.fixture
def mock_asset_manager(temp_dir):
    """Create a mock asset manager with test assets"""
    # Create test asset structure
    assets_dir = os.path.join(temp_dir, "assets")
    os.makedirs(os.path.join(assets_dir, "models", "props"), exist_ok=True)
    os.makedirs(os.path.join(assets_dir, "textures"), exist_ok=True)
    os.makedirs(os.path.join(assets_dir, "materials"), exist_ok=True)
    
    # Create dummy asset files
    test_files = [
        "models/props/chair.obj",
        "models/props/table.blend",
        "textures/wood_diffuse.png",
        "materials/wood_material.json"
    ]
    
    for file_path in test_files:
        full_path = os.path.join(assets_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write("Mock asset data")
    
    asset_manager = AssetManager()
    return asset_manager


@pytest.fixture
def test_cache(temp_dir):
    """Create a test cache instance"""
    cache_dir = os.path.join(temp_dir, "cache")
    return SmartCache(cache_dir=cache_dir, max_memory_mb=50)


@pytest.fixture
def prompt_engineer():
    """Create a prompt engineer for testing"""
    return PromptEngineer()


@pytest.fixture
def blender_studio(mock_config, mock_blender_executor, mock_gemini_api):
    """Create a complete Blender AI Studio instance for testing"""
    
    # Mock the BlenderExecutor in the main module
    with patch('main.BlenderExecutor', return_value=mock_blender_executor):
        studio = BlenderAIStudio()
        studio.blender_executor = mock_blender_executor
        yield studio


@pytest.fixture
def sample_scene_data():
    """Sample scene generation data for testing"""
    return {
        "prompt": "A modern office building in a city environment",
        "style": "realistic",
        "quality": "high",
        "scene_id": "test_scene_001",
        "requirements": {
            "scene_type": "outdoor",
            "complexity": "moderate",
            "elements": ["buildings", "street", "vehicles"],
            "style": "realistic"
        },
        "expected_commands": [
            {
                "type": "create_terrain",
                "params": {
                    "size": [100, 100],
                    "subdivision": 32,
                    "material": "concrete"
                }
            },
            {
                "type": "create_structure",
                "params": {
                    "structure_type": "building",
                    "position": [0, 0, 0],
                    "scale": [10, 10, 30],
                    "material": "glass_concrete"
                }
            }
        ]
    }


@pytest.fixture
def mock_file_system(temp_dir):
    """Create a mock file system with necessary directories"""
    # Create all necessary directories
    directories = [
        "output",
        "logs", 
        "cache",
        "assets/default/models",
        "assets/default/textures",
        "assets/default/materials",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(temp_dir, directory), exist_ok=True)
    
    # Create mock .env file
    env_content = f"""
GOOGLE_API_KEY=test_api_key
OUTPUT_DIRECTORY={temp_dir}/output
LOG_LEVEL=DEBUG
CACHE_SIZE_MB=100
DEVELOPMENT_MODE=true
TEST_MODE=true
"""
    
    with open(os.path.join(temp_dir, ".env"), 'w') as f:
        f.write(env_content)
    
    return temp_dir


# Test utilities
class AsyncContextManager:
    """Utility class for testing async context managers"""
    def __init__(self, return_value=None):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def create_mock_async_func(return_value=None, side_effect=None):
    """Create a mock async function"""
    async def mock_func(*args, **kwargs):
        if side_effect:
            if isinstance(side_effect, Exception):
                raise side_effect
            elif callable(side_effect):
                return side_effect(*args, **kwargs)
            else:
                return side_effect
        return return_value
    
    return Mock(side_effect=mock_func)


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Monitor for performance testing"""
    import time
    import psutil
    import threading
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_usage = []
            self.cpu_usage = []
            self.monitoring = False
            self._monitor_thread = None
        
        def start(self):
            self.start_time = time.time()
            self.monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor)
            self._monitor_thread.start()
        
        def stop(self):
            self.end_time = time.time()
            self.monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join()
        
        def _monitor(self):
            process = psutil.Process()
            while self.monitoring:
                try:
                    self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                    self.cpu_usage.append(process.cpu_percent())
                    time.sleep(0.1)
                except:
                    break
        
        def get_stats(self):
            return {
                "duration": self.end_time - self.start_time if self.end_time else 0,
                "max_memory_mb": max(self.memory_usage) if self.memory_usage else 0,
                "avg_memory_mb": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                "max_cpu_percent": max(self.cpu_usage) if self.cpu_usage else 0,
                "avg_cpu_percent": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
            }
    
    return PerformanceMonitor()


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "requires_blender: Tests requiring Blender installation")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_api: Tests requiring external API access")