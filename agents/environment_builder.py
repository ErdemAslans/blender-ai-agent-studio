"""Environment Builder Agent - Creates spatial compositions and structures"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentConfig


class Structure(BaseModel):
    """Represents a structure in the scene"""
    type: str = Field(..., description="Type of structure (building, terrain, etc.)")
    position: Tuple[float, float, float] = Field((0, 0, 0), description="3D position")
    rotation: Tuple[float, float, float] = Field((0, 0, 0), description="Rotation in degrees")
    scale: Tuple[float, float, float] = Field((1, 1, 1), description="Scale factors")
    material: str = Field("default", description="Material preset name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class TerrainConfig(BaseModel):
    """Configuration for terrain generation"""
    type: str = Field("flat", description="Terrain type (flat, hills, mountains, etc.)")
    size: Tuple[float, float] = Field((100, 100), description="Terrain dimensions")
    subdivision: int = Field(32, description="Mesh subdivision level")
    height_scale: float = Field(10.0, description="Maximum height variation")
    noise_scale: float = Field(0.5, description="Noise frequency")
    material: str = Field("terrain_default", description="Terrain material")


class EnvironmentLayout(BaseModel):
    """Complete environment layout specification"""
    terrain: Optional[TerrainConfig] = None
    structures: List[Structure] = Field(default_factory=list)
    boundaries: Tuple[float, float, float, float] = Field((-50, -50, 50, 50), description="Scene boundaries (min_x, min_y, max_x, max_y)")
    grid_size: float = Field(10.0, description="Grid size for structure placement")
    density: str = Field("medium", description="Structure density (sparse, medium, dense)")


class EnvironmentBuilderAgent(BaseAgent):
    """
    The Environment Builder Agent specializes in spatial composition and structural elements.
    Uses Blender's Python API to generate basic geometric structures, position buildings,
    create terrain, and establish the foundational layout.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="environment_builder",
                model="gemini-2.0-flash-exp",
                temperature=0.6,
                max_tokens=4096
            )
        super().__init__(config)
        
        # Environment presets
        self.environment_presets = {
            "city": {
                "structure_types": ["skyscraper", "office_building", "apartment", "shop"],
                "density": "dense",
                "grid_aligned": True,
                "terrain": "flat"
            },
            "forest": {
                "structure_types": ["tree", "rock", "bush", "fallen_log"],
                "density": "medium",
                "grid_aligned": False,
                "terrain": "hills"
            },
            "desert": {
                "structure_types": ["dune", "rock_formation", "cactus"],
                "density": "sparse",
                "grid_aligned": False,
                "terrain": "dunes"
            },
            "medieval": {
                "structure_types": ["castle", "house", "tower", "wall"],
                "density": "medium",
                "grid_aligned": False,
                "terrain": "hills"
            }
        }
        
        # Structure templates
        self.structure_templates = {
            "skyscraper": {
                "base_shape": "cube",
                "min_height": 50,
                "max_height": 200,
                "width_range": (10, 30),
                "material": "glass_steel"
            },
            "office_building": {
                "base_shape": "cube",
                "min_height": 20,
                "max_height": 60,
                "width_range": (15, 40),
                "material": "concrete_glass"
            },
            "tree": {
                "base_shape": "cylinder",
                "min_height": 5,
                "max_height": 25,
                "width_range": (0.5, 2),
                "material": "bark",
                "has_canopy": True
            }
        }
        
        self.layout_generation_prompt = """
        Generate a detailed environment layout for:
        Environment Type: {environment_type}
        Style: {style}
        Complexity: {complexity}
        Special Requirements: {requirements}
        
        Create a layout that includes:
        1. Terrain configuration appropriate for the environment
        2. List of structures with positions, types, and sizes
        3. Proper spacing and arrangement based on the environment type
        4. Consideration for the visual style
        
        Respond with a JSON object matching the EnvironmentLayout schema.
        """
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment building task"""
        task_params = input_data.get("parameters", {})
        environment_type = task_params.get("environment_type", "city")
        style = task_params.get("style", "realistic")
        complexity = task_params.get("complexity", "medium")
        quality = task_params.get("quality", "high")
        
        self.logger.info(f"Building {style} {environment_type} environment")
        
        # Generate environment layout
        layout = await self._generate_layout(
            environment_type, style, complexity, task_params
        )
        
        # Generate Blender commands
        blender_commands = self._generate_blender_commands(layout, quality)
        
        # Prepare output
        result = {
            "status": "success",
            "agent": self.config.name,
            "layout": layout.model_dump(),
            "blender_commands": blender_commands,
            "statistics": {
                "structure_count": len(layout.structures),
                "terrain_size": layout.terrain.size if layout.terrain else None,
                "scene_bounds": layout.boundaries
            }
        }
        
        self.update_metadata("generated_layout", layout.model_dump())
        
        return result
        
    async def _generate_layout(
        self, 
        environment_type: str, 
        style: str, 
        complexity: str,
        requirements: Dict[str, Any]
    ) -> EnvironmentLayout:
        """Generate environment layout using AI"""
        
        # Check if we have a preset for this environment type
        if environment_type in self.environment_presets:
            # Use preset as base
            preset = self.environment_presets[environment_type]
            layout = await self._generate_from_preset(preset, style, complexity)
        else:
            # Generate custom layout
            prompt = self.layout_generation_prompt.format(
                environment_type=environment_type,
                style=style,
                complexity=complexity,
                requirements=json.dumps(requirements)
            )
            
            response = await self.generate_response(prompt)
            
            try:
                layout_data = json.loads(response)
                layout = EnvironmentLayout(**layout_data)
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse layout: {e}")
                layout = self._create_default_layout(environment_type)
                
        return layout
        
    async def _generate_from_preset(
        self, 
        preset: Dict[str, Any], 
        style: str, 
        complexity: str
    ) -> EnvironmentLayout:
        """Generate layout from preset configuration"""
        
        # Determine scene size based on complexity
        size_multipliers = {"simple": 0.5, "medium": 1.0, "complex": 2.0}
        size_mult = size_multipliers.get(complexity, 1.0)
        
        # Set boundaries
        bound = 50 * size_mult
        boundaries = (-bound, -bound, bound, bound)
        
        # Create terrain
        terrain = self._create_terrain(preset["terrain"], boundaries)
        
        # Generate structures
        structures = self._generate_structures(
            preset["structure_types"],
            preset["density"],
            boundaries,
            preset["grid_aligned"],
            complexity
        )
        
        # Apply style modifications
        structures = self._apply_style_modifications(structures, style)
        
        return EnvironmentLayout(
            terrain=terrain,
            structures=structures,
            boundaries=boundaries,
            density=preset["density"]
        )
        
    def _create_terrain(
        self, 
        terrain_type: str, 
        boundaries: Tuple[float, float, float, float]
    ) -> TerrainConfig:
        """Create terrain configuration"""
        min_x, min_y, max_x, max_y = boundaries
        size = (max_x - min_x, max_y - min_y)
        
        terrain_configs = {
            "flat": TerrainConfig(
                type="flat",
                size=size,
                subdivision=16,
                height_scale=0.1,
                noise_scale=0.01
            ),
            "hills": TerrainConfig(
                type="hills",
                size=size,
                subdivision=64,
                height_scale=10.0,
                noise_scale=0.05
            ),
            "mountains": TerrainConfig(
                type="mountains",
                size=size,
                subdivision=128,
                height_scale=50.0,
                noise_scale=0.02
            ),
            "dunes": TerrainConfig(
                type="dunes",
                size=size,
                subdivision=64,
                height_scale=15.0,
                noise_scale=0.03,
                material="sand"
            )
        }
        
        return terrain_configs.get(terrain_type, terrain_configs["flat"])
        
    def _generate_structures(
        self,
        structure_types: List[str],
        density: str,
        boundaries: Tuple[float, float, float, float],
        grid_aligned: bool,
        complexity: str
    ) -> List[Structure]:
        """Generate structures based on parameters"""
        structures = []
        
        # Determine number of structures
        density_counts = {
            "sparse": {"simple": 5, "medium": 10, "complex": 20},
            "medium": {"simple": 15, "medium": 30, "complex": 60},
            "dense": {"simple": 30, "medium": 60, "complex": 120}
        }
        
        count = density_counts.get(density, density_counts["medium"]).get(complexity, 30)
        
        min_x, min_y, max_x, max_y = boundaries
        
        for i in range(count):
            # Select structure type
            structure_type = random.choice(structure_types)
            
            # Get template if available
            template = self.structure_templates.get(structure_type, {})
            
            # Generate position
            if grid_aligned:
                grid_size = 10
                x = random.randint(int(min_x/grid_size), int(max_x/grid_size)) * grid_size
                y = random.randint(int(min_y/grid_size), int(max_y/grid_size)) * grid_size
            else:
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                
            z = 0  # Will be adjusted based on terrain
            
            # Generate dimensions
            if template:
                height = random.uniform(
                    template.get("min_height", 5),
                    template.get("max_height", 20)
                )
                width_min, width_max = template.get("width_range", (5, 15))
                width = random.uniform(width_min, width_max)
                depth = width * random.uniform(0.8, 1.2)
                material = template.get("material", "default")
            else:
                height = random.uniform(5, 30)
                width = random.uniform(5, 20)
                depth = random.uniform(5, 20)
                material = "default"
                
            # Create structure
            structure = Structure(
                type=structure_type,
                position=(x, y, z),
                rotation=(0, random.uniform(0, 360), 0),
                scale=(width, depth, height),
                material=material,
                parameters={"template": structure_type}
            )
            
            structures.append(structure)
            
        return structures
        
    def _apply_style_modifications(self, structures: List[Structure], style: str) -> List[Structure]:
        """Apply style-specific modifications to structures"""
        style_mods = {
            "cyberpunk": {
                "height_multiplier": 1.5,
                "materials": ["neon_glass", "dark_metal", "holographic"],
                "add_glow": True
            },
            "post-apocalyptic": {
                "height_multiplier": 0.7,
                "materials": ["rusted_metal", "broken_concrete", "debris"],
                "add_damage": True
            },
            "fantasy": {
                "height_multiplier": 1.2,
                "materials": ["stone", "magic_crystal", "ancient_wood"],
                "add_magic": True
            }
        }
        
        if style in style_mods:
            mods = style_mods[style]
            
            for structure in structures:
                # Modify height
                scale = list(structure.scale)
                scale[2] *= mods["height_multiplier"]
                structure.scale = tuple(scale)
                
                # Change material
                if structure.material == "default":
                    structure.material = random.choice(mods["materials"])
                    
                # Add style-specific parameters
                for key, value in mods.items():
                    if key.startswith("add_"):
                        structure.parameters[key] = value
                        
        return structures
        
    def _generate_blender_commands(self, layout: EnvironmentLayout, quality: str) -> List[Dict[str, Any]]:
        """Generate Blender Python API commands"""
        commands = []
        
        # Clear scene command
        commands.append({
            "type": "clear_scene",
            "params": {}
        })
        
        # Create terrain
        if layout.terrain:
            commands.append({
                "type": "create_terrain",
                "params": {
                    "size": layout.terrain.size,
                    "subdivision": layout.terrain.subdivision,
                    "height_scale": layout.terrain.height_scale,
                    "noise_scale": layout.terrain.noise_scale,
                    "material": layout.terrain.material
                }
            })
            
        # Create structures
        for structure in layout.structures:
            command = {
                "type": "create_structure",
                "params": {
                    "structure_type": structure.type,
                    "position": structure.position,
                    "rotation": structure.rotation,
                    "scale": structure.scale,
                    "material": structure.material,
                    "parameters": structure.parameters
                }
            }
            commands.append(command)
            
        # Add quality settings
        commands.append({
            "type": "set_quality",
            "params": {"preset": quality}
        })
        
        return commands
        
    def _create_default_layout(self, environment_type: str) -> EnvironmentLayout:
        """Create a simple default layout as fallback"""
        return EnvironmentLayout(
            terrain=TerrainConfig(),
            structures=[
                Structure(
                    type="cube",
                    position=(0, 0, 5),
                    scale=(10, 10, 10)
                )
            ]
        )
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return "parameters" in input_data and isinstance(input_data["parameters"], dict)