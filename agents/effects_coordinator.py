"""Effects Coordinator Agent - Manages particles and weather effects"""

import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentConfig


class ParticleSystem(BaseModel):
    """Represents a particle system"""
    name: str = Field(..., description="Particle system name")
    type: str = Field(..., description="Particle type (emitter, hair, fluid)")
    emitter_location: Tuple[float, float, float] = Field((0, 0, 0), description="Emitter position")
    particle_count: int = Field(1000, description="Number of particles")
    lifetime: float = Field(5.0, description="Particle lifetime in seconds")
    physics_type: str = Field("newtonian", description="Physics simulation type")
    material: str = Field("default", description="Particle material")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class WeatherEffect(BaseModel):
    """Weather effect configuration"""
    type: str = Field(..., description="Weather type (rain, snow, fog, wind)")
    intensity: float = Field(0.5, ge=0, le=1, description="Effect intensity (0-1)")
    coverage: float = Field(1.0, ge=0, le=1, description="Scene coverage (0-1)")
    direction: Tuple[float, float, float] = Field((0, 0, -1), description="Primary direction vector")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Weather-specific parameters")


class AtmosphericEffect(BaseModel):
    """Atmospheric effect configuration"""
    type: str = Field(..., description="Atmospheric type (dust, smoke, steam, magical)")
    source_locations: List[Tuple[float, float, float]] = Field(default_factory=list, description="Effect source points")
    density: float = Field(0.1, ge=0, le=1, description="Effect density")
    color: Tuple[float, float, float] = Field((1, 1, 1), description="Effect color")
    animation_speed: float = Field(1.0, description="Animation speed multiplier")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Effect-specific parameters")


class EffectsSetup(BaseModel):
    """Complete effects setup specification"""
    particle_systems: List[ParticleSystem] = Field(default_factory=list)
    weather_effects: List[WeatherEffect] = Field(default_factory=list)
    atmospheric_effects: List[AtmosphericEffect] = Field(default_factory=list)
    global_physics: Dict[str, Any] = Field(default_factory=dict, description="Global physics settings")
    performance_settings: Dict[str, Any] = Field(default_factory=dict, description="Performance optimization settings")


class EffectsCoordinatorAgent(BaseAgent):
    """
    The Effects Coordinator Agent manages Blender's particle systems and 
    simulation tools to create weather effects, atmospheric elements, and 
    environmental dynamics. Configures rain systems, fog effects, and other 
    atmospheric enhancements using Blender's built-in capabilities.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="effects_coordinator",
                model="gemini-2.0-flash-exp",
                temperature=0.7,
                max_tokens=4096
            )
        super().__init__(config)
        
        # Weather effect presets
        self.weather_presets = {
            "rain": {
                "particle_count": 5000,
                "lifetime": 3.0,
                "velocity": (0, 0, -10),
                "size": 0.01,
                "material": "water_droplet",
                "physics": "newtonian",
                "gravity": -9.8
            },
            "snow": {
                "particle_count": 3000,
                "lifetime": 8.0,
                "velocity": (0.5, 0, -2),
                "size": 0.02,
                "material": "snow_flake",
                "physics": "newtonian",
                "gravity": -2.0
            },
            "fog": {
                "type": "volume",
                "density": 0.3,
                "material": "fog_volume",
                "animation": "drift",
                "coverage": 1.0
            },
            "dust": {
                "particle_count": 2000,
                "lifetime": 10.0,
                "velocity": (1, 1, 0.5),
                "size": 0.005,
                "material": "dust_particle",
                "physics": "boids"
            }
        }
        
        # Style-specific effect configurations
        self.style_effects = {
            "cyberpunk": {
                "steam_vents": {
                    "count": 5,
                    "color": (0.8, 0.9, 1.0),
                    "intensity": 0.7
                },
                "neon_glow": {
                    "bloom_intensity": 2.0,
                    "glow_color": (0, 1, 1)
                },
                "holographic_particles": {
                    "particle_count": 1000,
                    "material": "holographic",
                    "animation": "float"
                }
            },
            "post-apocalyptic": {
                "ash_fall": {
                    "particle_count": 8000,
                    "color": (0.3, 0.3, 0.3),
                    "lifetime": 15.0
                },
                "dust_clouds": {
                    "density": 0.4,
                    "color": (0.7, 0.6, 0.4),
                    "coverage": 0.8
                },
                "smoke_columns": {
                    "count": 3,
                    "height": 50,
                    "color": (0.2, 0.2, 0.2)
                }
            },
            "medieval": {
                "torch_smoke": {
                    "count": 10,
                    "color": (0.3, 0.3, 0.3),
                    "rise_speed": 2.0
                },
                "mist": {
                    "density": 0.15,
                    "color": (0.9, 0.9, 1.0),
                    "ground_level": True
                },
                "fireflies": {
                    "particle_count": 200,
                    "color": (1.0, 1.0, 0.5),
                    "flicker": True
                }
            },
            "scifi": {
                "energy_fields": {
                    "particle_count": 1500,
                    "color": (0, 0.8, 1),
                    "animation": "spiral"
                },
                "plasma_effects": {
                    "intensity": 0.8,
                    "color": (1, 0.5, 0),
                    "electrical": True
                }
            }
        }
        
        self.effects_design_prompt = """
        Design atmospheric and particle effects for:
        Weather: {weather}
        Atmosphere: {atmosphere}
        Style: {style}
        Environment: {environment_type}
        Special Requirements: {special_requirements}
        
        Create effects that enhance the scene's mood and atmosphere:
        1. Weather systems (rain, snow, fog) if specified
        2. Style-appropriate atmospheric effects
        3. Particle systems for ambiance
        4. Environmental dynamics (wind, movement)
        5. Performance-optimized settings
        
        Consider:
        - Visual impact and mood enhancement
        - Performance implications
        - Style consistency
        - Environmental appropriateness
        - Animation and movement patterns
        
        Respond with a JSON object matching the EffectsSetup schema.
        """
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process effects coordination task"""
        task_params = input_data.get("parameters", {})
        weather = task_params.get("weather")
        atmosphere = task_params.get("atmosphere", "clear")
        style = task_params.get("style", "realistic")
        
        # Get environment info from shared state
        shared_state = input_data.get("shared_state", {})
        environment_data = shared_state.get("metadata", {})
        environment_type = environment_data.get("environment_type", "generic")
        
        self.logger.info(f"Coordinating effects: weather={weather}, atmosphere={atmosphere}, style={style}")
        
        # Generate effects setup
        effects_setup = await self._generate_effects_setup(
            weather, atmosphere, style, environment_type, task_params
        )
        
        # Generate Blender commands
        blender_commands = self._generate_blender_commands(effects_setup)
        
        # Prepare output
        result = {
            "status": "success",
            "agent": self.config.name,
            "effects_setup": effects_setup.model_dump(),
            "blender_commands": blender_commands,
            "statistics": {
                "particle_systems": len(effects_setup.particle_systems),
                "weather_effects": len(effects_setup.weather_effects),
                "atmospheric_effects": len(effects_setup.atmospheric_effects),
                "total_particles": sum(ps.particle_count for ps in effects_setup.particle_systems)
            }
        }
        
        self.update_metadata("effects_setup", effects_setup.model_dump())
        
        return result
        
    async def _generate_effects_setup(
        self,
        weather: Optional[str],
        atmosphere: str,
        style: str,
        environment_type: str,
        special_requirements: Dict[str, Any]
    ) -> EffectsSetup:
        """Generate comprehensive effects setup"""
        
        effects_setup = EffectsSetup()
        
        # Add weather effects
        if weather:
            weather_effect = self._create_weather_effect(weather, style)
            if weather_effect:
                effects_setup.weather_effects.append(weather_effect)
                
                # Add corresponding particle systems
                particle_systems = self._create_weather_particles(weather, style)
                effects_setup.particle_systems.extend(particle_systems)
                
        # Add style-specific effects
        style_effects = self._create_style_effects(style, environment_type)
        effects_setup.atmospheric_effects.extend(style_effects)
        
        # Add atmospheric enhancement particles
        atmospheric_particles = self._create_atmospheric_particles(atmosphere, style)
        effects_setup.particle_systems.extend(atmospheric_particles)
        
        # Set performance settings
        effects_setup.performance_settings = self._optimize_performance_settings(
            len(effects_setup.particle_systems),
            sum(ps.particle_count for ps in effects_setup.particle_systems)
        )
        
        # Set global physics
        effects_setup.global_physics = {
            "gravity": -9.8,
            "air_resistance": 0.1,
            "wind_strength": 0.5 if weather in ["rain", "snow"] else 0.2
        }
        
        return effects_setup
        
    def _create_weather_effect(self, weather: str, style: str) -> Optional[WeatherEffect]:
        """Create weather effect configuration"""
        if weather not in self.weather_presets:
            return None
            
        intensity = 0.7 if style in ["post-apocalyptic", "dramatic"] else 0.5
        
        weather_configs = {
            "rain": WeatherEffect(
                type="rain",
                intensity=intensity,
                coverage=0.9,
                direction=(0.1, 0, -1),
                parameters={
                    "drop_size_variation": 0.3,
                    "wind_factor": 0.2,
                    "splash_effects": True
                }
            ),
            "snow": WeatherEffect(
                type="snow",
                intensity=intensity * 0.8,
                coverage=1.0,
                direction=(0.2, 0, -0.5),
                parameters={
                    "flake_variation": 0.5,
                    "accumulation": True,
                    "wind_drift": True
                }
            ),
            "fog": WeatherEffect(
                type="fog",
                intensity=intensity,
                coverage=1.0,
                direction=(0, 0, 0),
                parameters={
                    "density_variation": 0.3,
                    "drift_speed": 0.5,
                    "ground_fog": True
                }
            )
        }
        
        return weather_configs.get(weather)
        
    def _create_weather_particles(self, weather: str, style: str) -> List[ParticleSystem]:
        """Create particle systems for weather effects"""
        if weather not in self.weather_presets:
            return []
            
        preset = self.weather_presets[weather]
        particles = []
        
        if weather == "rain":
            # Main rain system
            rain_system = ParticleSystem(
                name="rain_main",
                type="emitter",
                emitter_location=(0, 0, 50),
                particle_count=preset["particle_count"],
                lifetime=preset["lifetime"],
                physics_type=preset["physics"],
                material=preset["material"],
                parameters={
                    "velocity": preset["velocity"],
                    "size": preset["size"],
                    "gravity": preset["gravity"],
                    "emit_from": "volume",
                    "volume_shape": "cube",
                    "volume_size": (100, 100, 10)
                }
            )
            particles.append(rain_system)
            
            # Add splash effects
            if style in ["cyberpunk", "realistic"]:
                splash_system = ParticleSystem(
                    name="rain_splash",
                    type="emitter",
                    emitter_location=(0, 0, 0),
                    particle_count=500,
                    lifetime=0.5,
                    physics_type="newtonian",
                    material="water_splash",
                    parameters={
                        "trigger_on_collision": True,
                        "velocity_random": 0.5
                    }
                )
                particles.append(splash_system)
                
        elif weather == "snow":
            snow_system = ParticleSystem(
                name="snow_main",
                type="emitter",
                emitter_location=(0, 0, 60),
                particle_count=preset["particle_count"],
                lifetime=preset["lifetime"],
                physics_type=preset["physics"],
                material=preset["material"],
                parameters={
                    "velocity": preset["velocity"],
                    "size": preset["size"],
                    "gravity": preset["gravity"],
                    "rotation_random": 1.0,
                    "size_random": 0.5
                }
            )
            particles.append(snow_system)
            
        elif weather == "dust":
            dust_system = ParticleSystem(
                name="dust_cloud",
                type="emitter",
                emitter_location=(0, 0, 5),
                particle_count=preset["particle_count"],
                lifetime=preset["lifetime"],
                physics_type=preset["physics"],
                material=preset["material"],
                parameters={
                    "velocity": preset["velocity"],
                    "size": preset["size"],
                    "boids_settings": {
                        "goal_strength": 0.1,
                        "separate_strength": 0.5
                    }
                }
            )
            particles.append(dust_system)
            
        return particles
        
    def _create_style_effects(self, style: str, environment_type: str) -> List[AtmosphericEffect]:
        """Create style-specific atmospheric effects"""
        if style not in self.style_effects:
            return []
            
        style_config = self.style_effects[style]
        effects = []
        
        if style == "cyberpunk":
            # Steam vents
            if "steam_vents" in style_config:
                steam_config = style_config["steam_vents"]
                effects.append(AtmosphericEffect(
                    type="steam",
                    source_locations=[(i * 20 - 40, 10, 0) for i in range(steam_config["count"])],
                    density=0.2,
                    color=steam_config["color"],
                    parameters={
                        "rise_speed": 3.0,
                        "dissipation_rate": 0.1,
                        "temperature_driven": True
                    }
                ))
                
        elif style == "post-apocalyptic":
            # Ash fall
            if "ash_fall" in style_config:
                ash_config = style_config["ash_fall"]
                effects.append(AtmosphericEffect(
                    type="ash",
                    source_locations=[(0, 0, 100)],
                    density=0.3,
                    color=ash_config["color"],
                    parameters={
                        "fall_speed": 1.0,
                        "wind_drift": True,
                        "accumulation": True
                    }
                ))
                
            # Smoke columns
            if "smoke_columns" in style_config:
                smoke_config = style_config["smoke_columns"]
                smoke_locations = [(i * 40 - 40, 20, 0) for i in range(smoke_config["count"])]
                effects.append(AtmosphericEffect(
                    type="smoke",
                    source_locations=smoke_locations,
                    density=0.4,
                    color=smoke_config["color"],
                    parameters={
                        "rise_height": smoke_config["height"],
                        "turbulence": 0.8,
                        "dissipation_height": smoke_config["height"] + 20
                    }
                ))
                
        elif style == "medieval":
            # Ground mist
            if "mist" in style_config:
                mist_config = style_config["mist"]
                effects.append(AtmosphericEffect(
                    type="mist",
                    source_locations=[(0, 0, -2)],
                    density=mist_config["density"],
                    color=mist_config["color"],
                    parameters={
                        "ground_hug": True,
                        "max_height": 5.0,
                        "drift_speed": 0.3
                    }
                ))
                
        elif style == "scifi":
            # Energy fields
            if "energy_fields" in style_config:
                energy_config = style_config["energy_fields"]
                effects.append(AtmosphericEffect(
                    type="energy",
                    source_locations=[(0, 0, 10), (30, 0, 10), (-30, 0, 10)],
                    density=0.1,
                    color=energy_config["color"],
                    animation_speed=2.0,
                    parameters={
                        "electrical_pattern": True,
                        "pulsation": 0.5,
                        "connection_lines": True
                    }
                ))
                
        return effects
        
    def _create_atmospheric_particles(self, atmosphere: str, style: str) -> List[ParticleSystem]:
        """Create atmospheric particle systems"""
        particles = []
        
        # Dust motes for realistic scenes
        if atmosphere in ["dusty", "hazy"] or style == "post-apocalyptic":
            dust_motes = ParticleSystem(
                name="dust_motes",
                type="emitter",
                emitter_location=(0, 0, 10),
                particle_count=500,
                lifetime=20.0,
                physics_type="boids",
                material="dust_mote",
                parameters={
                    "size": 0.002,
                    "velocity": (0.1, 0.1, 0.05),
                    "float_pattern": True,
                    "light_interaction": True
                }
            )
            particles.append(dust_motes)
            
        # Magical sparkles for fantasy
        if style == "fantasy" or atmosphere == "magical":
            sparkles = ParticleSystem(
                name="magic_sparkles",
                type="emitter",
                emitter_location=(0, 0, 5),
                particle_count=200,
                lifetime=5.0,
                physics_type="keyed",
                material="sparkle",
                parameters={
                    "size": 0.01,
                    "twinkle": True,
                    "color_variation": 0.3,
                    "random_motion": 0.5
                }
            )
            particles.append(sparkles)
            
        return particles
        
    def _optimize_performance_settings(self, system_count: int, total_particles: int) -> Dict[str, Any]:
        """Optimize performance settings based on complexity"""
        settings = {
            "viewport_display": "point" if total_particles > 10000 else "circle",
            "render_display": "object",
            "simplification": {
                "enabled": total_particles > 5000,
                "factor": 0.5 if total_particles > 10000 else 0.8
            },
            "cache_settings": {
                "memory_cache": min(1024, total_particles // 10),
                "disk_cache": total_particles > 15000
            }
        }
        
        return settings
        
    def _generate_blender_commands(self, setup: EffectsSetup) -> List[Dict[str, Any]]:
        """Generate Blender Python API commands for effects"""
        commands = []
        
        # Set global physics settings
        commands.append({
            "type": "setup_physics_world",
            "params": setup.global_physics
        })
        
        # Create particle systems
        for ps in setup.particle_systems:
            commands.append({
                "type": "create_particle_system",
                "params": {
                    "name": ps.name,
                    "type": ps.type,
                    "emitter_location": ps.emitter_location,
                    "particle_count": ps.particle_count,
                    "lifetime": ps.lifetime,
                    "physics_type": ps.physics_type,
                    "material": ps.material,
                    "parameters": ps.parameters
                }
            })
            
        # Create weather effects
        for weather in setup.weather_effects:
            commands.append({
                "type": "setup_weather",
                "params": {
                    "weather_type": weather.type,
                    "intensity": weather.intensity,
                    "coverage": weather.coverage,
                    "direction": weather.direction,
                    "parameters": weather.parameters
                }
            })
            
        # Create atmospheric effects
        for atm in setup.atmospheric_effects:
            commands.append({
                "type": "create_atmospheric_effect",
                "params": {
                    "effect_type": atm.type,
                    "source_locations": atm.source_locations,
                    "density": atm.density,
                    "color": atm.color,
                    "animation_speed": atm.animation_speed,
                    "parameters": atm.parameters
                }
            })
            
        # Set performance optimizations
        commands.append({
            "type": "optimize_effects_performance",
            "params": setup.performance_settings
        })
        
        return commands
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return "parameters" in input_data and isinstance(input_data["parameters"], dict)