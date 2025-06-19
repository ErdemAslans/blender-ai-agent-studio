"""Blender Python API executor for scene generation commands"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logging_config import get_logger


class BlenderExecutor:
    """Executes Blender operations through Python API or command line"""
    
    def __init__(self, blender_path: Optional[str] = None):
        self.logger = get_logger("blender_executor")
        self.blender_path = blender_path or self._find_blender()
        self.scene_file = None
        self.is_headless = True
        
    def _find_blender(self) -> str:
        """Find Blender executable automatically"""
        possible_paths = [
            "/usr/bin/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",
            "blender"  # Try PATH
        ]
        
        for path in possible_paths:
            if os.path.exists(path) or path == "blender":
                try:
                    result = subprocess.run([path, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.logger.info(f"Found Blender at: {path}")
                        return path
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
                    
        raise RuntimeError("Blender not found. Please install Blender or set BLENDER_PATH")
        
    def execute_commands(self, commands: List[Dict[str, Any]], output_file: str) -> Dict[str, Any]:
        """Execute a series of Blender commands"""
        try:
            # Create Python script for Blender
            script_content = self._generate_blender_script(commands, output_file)
            script_file = f"/tmp/blender_script_{os.getpid()}.py"
            
            with open(script_file, 'w') as f:
                f.write(script_content)
                
            # Execute Blender with script
            result = self._run_blender_script(script_file, output_file)
            
            # Clean up
            os.unlink(script_file)
            
            return {
                "status": "success" if result["returncode"] == 0 else "error",
                "output_file": output_file if result["returncode"] == 0 else None,
                "logs": result["stdout"],
                "errors": result["stderr"] if result["returncode"] != 0 else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute Blender commands: {e}")
            return {
                "status": "error",
                "output_file": None,
                "logs": "",
                "errors": str(e)
            }
            
    def _generate_blender_script(self, commands: List[Dict[str, Any]], output_file: str) -> str:
        """Generate Python script for Blender execution"""
        script = """
import bpy
import bmesh
import mathutils
import os
import sys
from mathutils import Vector, Euler
import random

# Clear existing scene
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
        
    # Clear lights (except default if needed)
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

def create_terrain(size, subdivision, height_scale, noise_scale, material="terrain_default"):
    \"\"\"Create terrain mesh\"\"\"
    bpy.ops.mesh.primitive_plane_add(size=max(size), location=(0, 0, 0))
    terrain = bpy.context.active_object
    terrain.name = "Terrain"
    
    # Add subdivision
    modifier = terrain.modifiers.new(name="Subdivision", type='SUBSURF')
    modifier.levels = min(subdivision // 32, 4)  # Limit subdivision
    
    # Add displacement for height variation
    if height_scale > 0:
        displacement = terrain.modifiers.new(name="Displacement", type='DISPLACE')
        displacement.strength = height_scale
        displacement.mid_level = 0
        
        # Create noise texture
        noise_texture = bpy.data.textures.new(name="TerrainNoise", type='CLOUDS')
        noise_texture.noise_scale = noise_scale
        displacement.texture = noise_texture
    
    # Apply modifiers
    bpy.context.view_layer.objects.active = terrain
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    if height_scale > 0:
        bpy.ops.object.modifier_apply(modifier="Displacement")
    
    return terrain

def create_structure(structure_type, position, rotation, scale, material, parameters):
    \"\"\"Create a structure object\"\"\"
    location = Vector(position)
    rot = Euler([r * 3.14159/180 for r in rotation], 'XYZ')
    
    if structure_type == "cube" or structure_type == "building":
        bpy.ops.mesh.primitive_cube_add(location=location, rotation=rot, scale=scale)
    elif structure_type == "cylinder" or structure_type == "tree":
        bpy.ops.mesh.primitive_cylinder_add(location=location, rotation=rot, scale=scale)
    elif structure_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(location=location, rotation=rot, scale=scale)
    else:
        # Default to cube
        bpy.ops.mesh.primitive_cube_add(location=location, rotation=rot, scale=scale)
    
    obj = bpy.context.active_object
    obj.name = f"{structure_type}_{len(bpy.data.objects)}"
    
    # Add material
    if material != "default":
        create_material(obj, material)
    
    return obj

def create_light(light_type, position, rotation, color, energy, size, parameters):
    \"\"\"Create a light object\"\"\"
    location = Vector(position)
    rot = Euler([r * 3.14159/180 for r in rotation], 'XYZ')
    
    bpy.ops.object.light_add(type=light_type.upper(), location=location, rotation=rot)
    light_obj = bpy.context.active_object
    light = light_obj.data
    
    light.energy = energy
    light.color = color
    
    if hasattr(light, 'size'):
        light.size = size
    if hasattr(light, 'shadow_soft_size'):
        light.shadow_soft_size = size
        
    # Handle special parameters
    if 'cone_angle' in parameters and light_type == 'spot':
        light.spot_size = parameters['cone_angle'] * 3.14159/180
        
    return light_obj

def create_material(obj, material_name):
    \"\"\"Create and assign material to object\"\"\"
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    nodes.clear()
    
    # Create basic material based on name
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    if "glass" in material_name.lower():
        bsdf = nodes.new(type='ShaderNodeBsdfGlass')
        bsdf.inputs['IOR'].default_value = 1.45
    elif "metal" in material_name.lower():
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Roughness'].default_value = 0.1
    elif "neon" in material_name.lower():
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = (0, 1, 1, 1)
        emission.inputs['Strength'].default_value = 5.0
        
        # Mix emission with BSDF
        mix = nodes.new(type='ShaderNodeMixShader')
        mat.node_tree.links.new(emission.outputs['Emission'], mix.inputs[1])
        mat.node_tree.links.new(bsdf.outputs['BSDF'], mix.inputs[2])
        mat.node_tree.links.new(mix.outputs['Shader'], output.inputs['Surface'])
        
        obj.data.materials.append(mat)
        return
    else:
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    obj.data.materials.append(mat)

def create_particle_system(name, type, emitter_location, particle_count, lifetime, physics_type, material, parameters):
    \"\"\"Create particle system\"\"\"
    # Create emitter object
    bpy.ops.mesh.primitive_cube_add(location=emitter_location, scale=(0.1, 0.1, 0.1))
    emitter = bpy.context.active_object
    emitter.name = f"Emitter_{name}"
    
    # Add particle system
    bpy.ops.object.particle_system_add()
    ps = emitter.particle_systems[0]
    ps.name = name
    
    settings = ps.settings
    settings.count = min(particle_count, 10000)  # Limit for performance
    settings.lifetime = lifetime
    settings.emit_from = 'VOLUME'
    
    # Set physics type
    if physics_type == 'newtonian':
        settings.physics_type = 'NEWTON'
    elif physics_type == 'boids':
        settings.physics_type = 'BOIDS'
    else:
        settings.physics_type = 'NEWTON'
    
    # Apply parameters
    if 'velocity' in parameters:
        vel = parameters['velocity']
        settings.normal_factor = vel[2] if len(vel) > 2 else 1.0
        
    if 'size' in parameters:
        settings.particle_size = parameters['size']
        
    return emitter

def setup_render_settings(quality_preset="high"):
    \"\"\"Setup render settings\"\"\"
    scene = bpy.context.scene
    
    if quality_preset == "draft":
        scene.render.resolution_x = 960
        scene.render.resolution_y = 540
        scene.cycles.samples = 32
    elif quality_preset == "preview":
        scene.render.resolution_x = 1440
        scene.render.resolution_y = 810
        scene.cycles.samples = 64
    elif quality_preset == "high":
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.cycles.samples = 128
    else:  # ultra
        scene.render.resolution_x = 3840
        scene.render.resolution_y = 2160
        scene.cycles.samples = 256
    
    # Set render engine to Cycles
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU' if bpy.context.preferences.addons['cycles'].preferences.compute_device_type == 'CUDA' else 'CPU'

def save_scene(filepath):
    \"\"\"Save the scene\"\"\"
    bpy.ops.wm.save_as_mainfile(filepath=filepath)

# Main execution
try:
    print("Starting Blender scene generation...")
    
    # Clear scene first
    clear_scene()
    
    # Execute commands
"""
        
        # Add command execution
        for i, command in enumerate(commands):
            cmd_type = command.get("type", "unknown")
            params = command.get("params", {})
            
            if cmd_type == "clear_scene":
                script += "    clear_scene()\n"
            elif cmd_type == "create_terrain":
                script += f"    create_terrain({params.get('size', (100, 100))}, {params.get('subdivision', 32)}, {params.get('height_scale', 0)}, {params.get('noise_scale', 0.1)}, '{params.get('material', 'terrain_default')}')\n"
            elif cmd_type == "create_structure":
                script += f"    create_structure('{params.get('structure_type', 'cube')}', {params.get('position', (0, 0, 0))}, {params.get('rotation', (0, 0, 0))}, {params.get('scale', (1, 1, 1))}, '{params.get('material', 'default')}', {params.get('parameters', {})})\n"
            elif cmd_type == "create_light":
                script += f"    create_light('{params.get('light_type', 'sun')}', {params.get('position', (0, 0, 10))}, {params.get('rotation', (0, 0, 0))}, {params.get('color', (1, 1, 1))}, {params.get('energy', 1000)}, {params.get('size', 0.1)}, {params.get('parameters', {})})\n"
            elif cmd_type == "create_particle_system":
                script += f"    create_particle_system('{params.get('name', 'particles')}', '{params.get('type', 'emitter')}', {params.get('emitter_location', (0, 0, 10))}, {params.get('particle_count', 1000)}, {params.get('lifetime', 5.0)}, '{params.get('physics_type', 'newtonian')}', '{params.get('material', 'default')}', {params.get('parameters', {})})\n"
            elif cmd_type == "set_quality":
                script += f"    setup_render_settings('{params.get('preset', 'high')}')\n"
        
        # Add scene saving
        script += f"""
    # Setup final render settings
    setup_render_settings()
    
    # Save the scene
    save_scene(r"{output_file}")
    
    print("Scene generation completed successfully!")
    
except Exception as e:
    print(f"Error during scene generation: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        return script
        
    def _run_blender_script(self, script_file: str, output_file: str) -> Dict[str, Any]:
        """Run Blender with the generated script"""
        cmd = [
            self.blender_path,
            "--background",
            "--python", script_file
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": "Blender execution timed out"
            }
        except Exception as e:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Failed to run Blender: {str(e)}"
            }
            
    def render_scene(self, scene_file: str, output_image: str) -> Dict[str, Any]:
        """Render a scene to an image"""
        render_script = f"""
import bpy

# Open the scene file
bpy.ops.wm.open_mainfile(filepath=r"{scene_file}")

# Set output path
bpy.context.scene.render.filepath = r"{output_image}"

# Render
bpy.ops.render.render(write_still=True)

print("Render completed!")
"""
        
        script_file = f"/tmp/render_script_{os.getpid()}.py"
        with open(script_file, 'w') as f:
            f.write(render_script)
            
        try:
            result = self._run_blender_script(script_file, scene_file)
            os.unlink(script_file)
            
            return {
                "status": "success" if result["returncode"] == 0 else "error",
                "output_image": output_image if result["returncode"] == 0 else None,
                "logs": result["stdout"],
                "errors": result["stderr"] if result["returncode"] != 0 else None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "output_image": None,
                "logs": "",
                "errors": str(e)
            }