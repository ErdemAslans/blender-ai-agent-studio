# Default Asset Library

This directory contains the default asset library for the Blender AI Agent Studio. Assets are organized by category and type for efficient management and retrieval.

## Directory Structure

```
assets/default/
├── models/
│   ├── architecture/     # Buildings, structures, architectural elements
│   ├── props/           # Furniture, objects, decorative items
│   ├── vehicles/        # Cars, trucks, transportation
│   └── nature/          # Trees, rocks, natural elements
├── textures/
│   ├── surfaces/        # Wall, floor, ground textures
│   ├── materials/       # Wood, metal, fabric textures
│   └── patterns/        # Decorative patterns and overlays
└── materials/
    ├── pbr/             # PBR material definitions
    ├── procedural/      # Procedural material setups
    └── presets/         # Pre-configured material presets
```

## Supported Formats

### 3D Models
- `.blend` - Native Blender files (preferred)
- `.obj` - Wavefront OBJ with `.mtl` materials
- `.fbx` - Autodesk FBX format
- `.gltf/.glb` - glTF 2.0 format
- `.dae` - COLLADA format

### Textures
- `.png` - PNG images (preferred for color/normal/roughness)
- `.jpg/.jpeg` - JPEG images (for diffuse textures)
- `.exr` - OpenEXR (for HDR textures)
- `.hdr` - HDR format (for environment lighting)

### Materials
- `.json` - Material definitions in JSON format
- `.yaml` - Material definitions in YAML format

## Asset Naming Convention

Assets should follow this naming convention for optimal organization:

```
[category]_[style]_[name]_[variant].[extension]

Examples:
- architecture_modern_building_01.blend
- prop_medieval_chair_wooden.obj
- texture_surface_brick_weathered.png
- material_metal_steel_brushed.json
```

## Style Tags

Assets can be tagged with style indicators:
- `modern` - Contemporary, current designs
- `medieval` - Historical, fantasy medieval style
- `cyberpunk` - Futuristic, neon, high-tech
- `industrial` - Factory, mechanical, utilitarian
- `nature` - Organic, natural, environmental
- `cartoon` - Stylized, low-poly, non-realistic
- `realistic` - Photorealistic, high detail

## Quality Guidelines

### Low Poly (Performance)
- Models: < 1,000 triangles
- Textures: 512x512 or 1024x1024
- Use: Background elements, mobile targets

### Medium Poly (Balanced)
- Models: 1,000 - 10,000 triangles
- Textures: 1024x1024 or 2048x2048
- Use: Mid-ground objects, standard scenes

### High Poly (Quality)
- Models: 10,000+ triangles
- Textures: 2048x2048 or 4096x4096
- Use: Hero objects, close-up detail

## Adding New Assets

1. Place assets in the appropriate category directory
2. Follow naming conventions
3. Include metadata in filename or separate `.json` file
4. Ensure proper UV mapping and materials
5. Test asset in Blender before committing

## Asset Metadata

For complex assets, include a `.json` metadata file:

```json
{
  "name": "Modern Office Building",
  "description": "Contemporary office building with glass facade",
  "category": "architecture",
  "style_tags": ["modern", "urban", "glass"],
  "polygon_count": 15420,
  "materials": ["glass", "concrete", "steel"],
  "license": "CC0",
  "author": "Asset Creator",
  "version": "1.0"
}
```

## License

All assets in this directory should be properly licensed. Default assumption is proprietary unless otherwise specified. Supported licenses:
- `CC0` - Creative Commons Zero (public domain)
- `CC-BY` - Creative Commons Attribution
- `proprietary` - All rights reserved
- `custom` - See individual asset license

## Performance Notes

- Keep polygon counts reasonable for real-time use
- Use texture atlases when possible
- Optimize materials for Cycles/Eevee rendering
- Consider LOD (Level of Detail) variants for complex models
- Test performance impact in target scenes