"""Lighting Designer Agent - Manages illumination and atmospheric mood"""

import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentConfig


class Light(BaseModel):
    """Represents a light in the scene"""
    type: str = Field(..., description="Light type (sun, point, spot, area)")
    position: Tuple[float, float, float] = Field((0, 0, 10), description="3D position")
    rotation: Tuple[float, float, float] = Field((0, 0, 0), description="Rotation in degrees")
    color: Tuple[float, float, float] = Field((1, 1, 1), description="RGB color (0-1)")
    energy: float = Field(1000, description="Light intensity")
    size: float = Field(0.1, description="Light size (for soft shadows)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class MaterialEmission(BaseModel):
    """Emission settings for materials"""
    object_name: str = Field(..., description="Name of object to apply emission to")
    color: Tuple[float, float, float] = Field((1, 1, 1), description="Emission color")
    strength: float = Field(1.0, description="Emission strength")


class AtmosphereSettings(BaseModel):
    """Atmospheric and environment settings"""
    fog_density: float = Field(0.0, description="Volumetric fog density")
    fog_color: Tuple[float, float, float] = Field((0.5, 0.5, 0.5), description="Fog color")
    ambient_strength: float = Field(0.1, description="Ambient light strength")
    ambient_color: Tuple[float, float, float] = Field((0.1, 0.1, 0.2), description="Ambient color")
    bloom_intensity: float = Field(0.0, description="Bloom/glow effect intensity")
    exposure: float = Field(1.0, description="Camera exposure")


class LightingSetup(BaseModel):
    """Complete lighting setup specification"""
    lights: List[Light] = Field(default_factory=list)
    material_emissions: List[MaterialEmission] = Field(default_factory=list)
    atmosphere: AtmosphereSettings = Field(default_factory=AtmosphereSettings)
    hdri_path: Optional[str] = Field(None, description="HDRI environment texture path")
    color_temperature: float = Field(6500, description="Overall color temperature in Kelvin")


class LightingDesignerAgent(BaseAgent):
    """
    The Lighting Designer Agent focuses exclusively on illumination and mood creation.
    Configures Blender's lighting systems, sets up material emission properties for 
    neon effects, adjusts color temperatures, and creates atmospheric lighting conditions.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="lighting_designer",
                model="gemini-2.0-flash-exp",
                temperature=0.7,
                max_tokens=4096
            )
        super().__init__(config)
        
        # Lighting presets for different times of day
        self.time_presets = {
            "day": {
                "sun_energy": 5.0,
                "sun_angle": 45,
                "sun_color": (1.0, 0.95, 0.8),
                "ambient_strength": 0.3,
                "fog_density": 0.01
            },
            "night": {
                "sun_energy": 0.1,
                "sun_angle": -30,
                "sun_color": (0.6, 0.7, 1.0),
                "ambient_strength": 0.05,
                "fog_density": 0.02
            },
            "dawn": {
                "sun_energy": 2.0,
                "sun_angle": 5,
                "sun_color": (1.0, 0.6, 0.4),
                "ambient_strength": 0.15,
                "fog_density": 0.03
            },
            "dusk": {
                "sun_energy": 3.0,
                "sun_angle": 15,
                "sun_color": (1.0, 0.5, 0.3),
                "ambient_strength": 0.2,
                "fog_density": 0.02
            }
        }
        
        # Style-specific lighting configurations
        self.style_configs = {
            "cyberpunk": {
                "primary_colors": [(0, 1, 1), (1, 0, 1), (1, 1, 0)],  # Cyan, Magenta, Yellow
                "emission_strength": 5.0,
                "fog_density": 0.3,
                "bloom_intensity": 1.5,
                "contrast_ratio": 0.8
            },
            "medieval": {
                "primary_colors": [(1, 0.8, 0.4), (1, 0.6, 0.2)],  # Warm torch colors
                "emission_strength": 2.0,
                "fog_density": 0.1,
                "bloom_intensity": 0.3,
                "contrast_ratio": 0.5
            },
            "post-apocalyptic": {
                "primary_colors": [(0.8, 0.6, 0.4), (0.5, 0.5, 0.5)],  # Dusty, desaturated
                "emission_strength": 1.0,
                "fog_density": 0.4,
                "bloom_intensity": 0.2,
                "contrast_ratio": 0.3
            },
            "scifi": {
                "primary_colors": [(0, 0.8, 1), (0.5, 1, 0.5), (1, 0.5, 0)],  # Cool tech colors
                "emission_strength": 3.0,
                "fog_density": 0.15,
                "bloom_intensity": 1.0,
                "contrast_ratio": 0.7
            }
        }
        
        self.lighting_design_prompt = """
        Design a lighting setup for:
        Time of Day: {time_of_day}
        Mood: {mood}
        Style: {style}
        Environment Type: {environment_type}
        Special Requirements: {special_requirements}
        
        Create a lighting design that includes:
        1. Key lights (sun/main directional light)
        2. Fill lights to reduce harsh shadows
        3. Accent/rim lights for depth
        4. Practical lights (lamps, neon signs, etc.) based on style
        5. Atmospheric settings (fog, ambient, bloom)
        6. Material emissions for style-appropriate elements
        
        Consider the mood and create appropriate:
        - Color temperatures
        - Light intensities
        - Shadow softness
        - Atmospheric effects
        
        Respond with a JSON object matching the LightingSetup schema.
        """
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process lighting design task"""
        task_params = input_data.get("parameters", {})
        time_of_day = task_params.get("time_of_day", "day")
        mood = task_params.get("mood", "neutral")
        style = task_params.get("style", "realistic")
        special_requirements = task_params.get("special_requirements", "")
        
        # Get environment info from shared state if available
        shared_state = input_data.get("shared_state", {})
        environment_type = shared_state.get("metadata", {}).get("environment_type", "generic")
        
        self.logger.info(f"Designing {mood} lighting for {style} {time_of_day} scene")
        
        # Generate lighting setup
        lighting_setup = await self._generate_lighting_setup(
            time_of_day, mood, style, environment_type, special_requirements
        )
        
        # Generate Blender commands
        blender_commands = self._generate_blender_commands(lighting_setup)
        
        # Prepare output
        result = {
            "status": "success",
            "agent": self.config.name,
            "lighting_setup": lighting_setup.model_dump(),
            "blender_commands": blender_commands,
            "statistics": {
                "light_count": len(lighting_setup.lights),
                "emission_count": len(lighting_setup.material_emissions),
                "has_fog": lighting_setup.atmosphere.fog_density > 0,
                "has_bloom": lighting_setup.atmosphere.bloom_intensity > 0
            }
        }
        
        self.update_metadata("lighting_setup", lighting_setup.model_dump())
        
        return result
        
    async def _generate_lighting_setup(
        self,
        time_of_day: str,
        mood: str,
        style: str,
        environment_type: str,
        special_requirements: str
    ) -> LightingSetup:
        """Generate lighting setup using AI and presets"""
        
        # Start with time-based preset
        base_setup = self._create_base_setup(time_of_day)
        
        # Apply style modifications
        style_setup = self._apply_style_lighting(base_setup, style, environment_type)
        
        # Adjust for mood
        mood_setup = await self._adjust_for_mood(style_setup, mood, special_requirements)
        
        return mood_setup
        
    def _create_base_setup(self, time_of_day: str) -> LightingSetup:
        """Create base lighting setup from time preset"""
        preset = self.time_presets.get(time_of_day, self.time_presets["day"])
        
        # Create sun light
        sun_light = Light(
            type="sun",
            position=(0, 0, 100),
            rotation=(preset["sun_angle"], 45, 0),
            color=preset["sun_color"],
            energy=preset["sun_energy"],
            size=0.1
        )
        
        # Create basic atmosphere
        atmosphere = AtmosphereSettings(
            fog_density=preset["fog_density"],
            ambient_strength=preset["ambient_strength"],
            ambient_color=(0.4, 0.5, 0.7) if time_of_day == "night" else (0.7, 0.8, 1.0)
        )
        
        return LightingSetup(
            lights=[sun_light],
            atmosphere=atmosphere
        )
        
    def _apply_style_lighting(
        self,
        base_setup: LightingSetup,
        style: str,
        environment_type: str
    ) -> LightingSetup:
        """Apply style-specific lighting modifications"""
        
        if style not in self.style_configs:
            return base_setup
            
        style_config = self.style_configs[style]
        
        # Update atmosphere
        base_setup.atmosphere.fog_density = style_config["fog_density"]
        base_setup.atmosphere.bloom_intensity = style_config["bloom_intensity"]
        
        # Add style-specific lights
        if style == "cyberpunk":
            # Add neon accent lights
            for i in range(5):
                color = style_config["primary_colors"][i % len(style_config["primary_colors"])]
                base_setup.lights.append(Light(
                    type="area",
                    position=(i * 20 - 40, 10, 15),
                    color=color,
                    energy=2000,
                    size=5,
                    parameters={"neon": True}
                ))
                
            # Add emission to buildings
            base_setup.material_emissions.extend([
                MaterialEmission(
                    object_name="building_*",
                    color=style_config["primary_colors"][0],
                    strength=style_config["emission_strength"]
                ),
                MaterialEmission(
                    object_name="sign_*",
                    color=style_config["primary_colors"][1],
                    strength=style_config["emission_strength"] * 2
                )
            ])
            
        elif style == "medieval":
            # Add torch lights
            torch_positions = [(-20, -20, 5), (20, -20, 5), (-20, 20, 5), (20, 20, 5)]
            for pos in torch_positions:
                base_setup.lights.append(Light(
                    type="point",
                    position=pos,
                    color=style_config["primary_colors"][0],
                    energy=500,
                    size=0.5,
                    parameters={"flicker": True}
                ))
                
        elif style == "post-apocalyptic":
            # Reduce main light intensity
            if base_setup.lights:
                base_setup.lights[0].energy *= 0.6
                
            # Add dusty atmosphere
            base_setup.atmosphere.fog_color = (0.7, 0.6, 0.5)
            
        elif style == "scifi":
            # Add tech lights
            for i in range(3):
                base_setup.lights.append(Light(
                    type="spot",
                    position=(i * 30 - 30, 0, 30),
                    rotation=(-45, 0, 0),
                    color=style_config["primary_colors"][i],
                    energy=3000,
                    size=0.2,
                    parameters={"cone_angle": 45}
                ))
                
        return base_setup
        
    async def _adjust_for_mood(
        self,
        setup: LightingSetup,
        mood: str,
        special_requirements: str
    ) -> LightingSetup:
        """Adjust lighting for specific mood"""
        
        mood_adjustments = {
            "dramatic": {
                "contrast_multiplier": 1.5,
                "key_light_multiplier": 1.3,
                "fill_light_multiplier": 0.5,
                "rim_light": True
            },
            "mysterious": {
                "contrast_multiplier": 1.2,
                "fog_multiplier": 2.0,
                "ambient_multiplier": 0.5,
                "color_shift": (-0.1, 0, 0.2)  # Shift to blue
            },
            "cheerful": {
                "contrast_multiplier": 0.8,
                "key_light_multiplier": 1.2,
                "ambient_multiplier": 1.5,
                "color_shift": (0.1, 0.1, 0)  # Warmer
            },
            "ominous": {
                "contrast_multiplier": 1.8,
                "key_light_multiplier": 0.7,
                "fog_multiplier": 1.5,
                "color_shift": (-0.2, -0.2, -0.1)  # Darker, desaturated
            }
        }
        
        if mood in mood_adjustments:
            adj = mood_adjustments[mood]
            
            # Adjust main lights
            for light in setup.lights:
                if light.type == "sun":
                    light.energy *= adj.get("key_light_multiplier", 1.0)
                    
            # Adjust atmosphere
            setup.atmosphere.fog_density *= adj.get("fog_multiplier", 1.0)
            setup.atmosphere.ambient_strength *= adj.get("ambient_multiplier", 1.0)
            
            # Apply color shift
            if "color_shift" in adj:
                shift = adj["color_shift"]
                for light in setup.lights:
                    light.color = tuple(
                        max(0, min(1, c + s))
                        for c, s in zip(light.color, shift)
                    )
                    
            # Add rim light for dramatic mood
            if adj.get("rim_light") and mood == "dramatic":
                setup.lights.append(Light(
                    type="area",
                    position=(0, -50, 20),
                    rotation=(30, 0, 0),
                    color=(0.8, 0.9, 1.0),
                    energy=3000,
                    size=10,
                    parameters={"rim_light": True}
                ))
                
        # Handle special requirements
        if special_requirements:
            prompt = f"""
            Current lighting setup has {len(setup.lights)} lights with {mood} mood.
            Additional requirements: {special_requirements}
            
            Suggest specific adjustments to meet these requirements.
            Keep response brief and actionable.
            """
            
            adjustments = await self.generate_response(prompt)
            self.logger.info(f"Special lighting adjustments: {adjustments}")
            
        return setup
        
    def _generate_blender_commands(self, setup: LightingSetup) -> List[Dict[str, Any]]:
        """Generate Blender Python API commands for lighting"""
        commands = []
        
        # Clear existing lights
        commands.append({
            "type": "clear_lights",
            "params": {}
        })
        
        # Create lights
        for light in setup.lights:
            commands.append({
                "type": "create_light",
                "params": {
                    "light_type": light.type,
                    "position": light.position,
                    "rotation": light.rotation,
                    "color": light.color,
                    "energy": light.energy,
                    "size": light.size,
                    "parameters": light.parameters
                }
            })
            
        # Set up material emissions
        for emission in setup.material_emissions:
            commands.append({
                "type": "set_emission",
                "params": {
                    "object_pattern": emission.object_name,
                    "color": emission.color,
                    "strength": emission.strength
                }
            })
            
        # Configure atmosphere
        commands.append({
            "type": "setup_atmosphere",
            "params": {
                "fog_density": setup.atmosphere.fog_density,
                "fog_color": setup.atmosphere.fog_color,
                "ambient_strength": setup.atmosphere.ambient_strength,
                "ambient_color": setup.atmosphere.ambient_color
            }
        })
        
        # Set up post-processing
        if setup.atmosphere.bloom_intensity > 0:
            commands.append({
                "type": "enable_bloom",
                "params": {
                    "intensity": setup.atmosphere.bloom_intensity,
                    "threshold": 1.0,
                    "radius": 6.0
                }
            })
            
        # Set exposure
        commands.append({
            "type": "set_exposure",
            "params": {"value": setup.atmosphere.exposure}
        })
        
        # Set HDRI if specified
        if setup.hdri_path:
            commands.append({
                "type": "set_hdri",
                "params": {"path": setup.hdri_path}
            })
            
        return commands
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return "parameters" in input_data and isinstance(input_data["parameters"], dict)