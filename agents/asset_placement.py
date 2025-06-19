"""Asset Placement Agent - Handles intelligent object positioning"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentConfig


class AssetInstance(BaseModel):
    """Represents an asset instance in the scene"""
    asset_name: str = Field(..., description="Name of the asset")
    category: str = Field(..., description="Asset category (vehicle, prop, nature, etc.)")
    position: Tuple[float, float, float] = Field((0, 0, 0), description="3D position")
    rotation: Tuple[float, float, float] = Field((0, 0, 0), description="Rotation in degrees")
    scale: Tuple[float, float, float] = Field((1, 1, 1), description="Scale factors")
    variation: Optional[str] = Field(None, description="Asset variation/version")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class PlacementRule(BaseModel):
    """Rules for asset placement"""
    category: str = Field(..., description="Asset category this rule applies to")
    surface_types: List[str] = Field(default_factory=list, description="Valid surface types")
    min_distance: float = Field(0, description="Minimum distance from other objects")
    max_slope: float = Field(45, description="Maximum slope angle in degrees")
    avoid_overlap: bool = Field(True, description="Avoid overlapping with other objects")
    clustering: bool = Field(False, description="Allow clustering of similar objects")
    density_per_area: float = Field(1.0, description="Objects per unit area")


class AssetLibrary(BaseModel):
    """Asset library configuration"""
    name: str = Field(..., description="Library name")
    path: str = Field(..., description="Path to asset files")
    categories: Dict[str, List[str]] = Field(default_factory=dict, description="Available assets by category")
    style_tags: List[str] = Field(default_factory=list, description="Style tags for this library")


class PlacementPlan(BaseModel):
    """Complete asset placement plan"""
    assets: List[AssetInstance] = Field(default_factory=list)
    rules: List[PlacementRule] = Field(default_factory=list)
    libraries: List[str] = Field(default_factory=list, description="Used asset libraries")
    statistics: Dict[str, int] = Field(default_factory=dict, description="Placement statistics")


class AssetPlacementAgent(BaseAgent):
    """
    The Asset Placement Agent handles object positioning and scene population.
    Works with existing 3D model libraries and applies intelligent placement 
    algorithms to position cars, buildings, street furniture, and environmental 
    details in contextually appropriate locations.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="asset_placement",
                model="gemini-2.0-flash-exp",
                temperature=0.6,
                max_tokens=4096
            )
        super().__init__(config)
        
        # Asset categories and their typical placement rules
        self.default_placement_rules = {
            "vehicles": PlacementRule(
                category="vehicles",
                surface_types=["road", "street", "parking"],
                min_distance=3.0,
                max_slope=15,
                avoid_overlap=True,
                clustering=False,
                density_per_area=0.1
            ),
            "street_furniture": PlacementRule(
                category="street_furniture",
                surface_types=["sidewalk", "plaza", "street"],
                min_distance=2.0,
                max_slope=10,
                avoid_overlap=True,
                clustering=True,
                density_per_area=0.2
            ),
            "vegetation": PlacementRule(
                category="vegetation",
                surface_types=["ground", "dirt", "grass"],
                min_distance=1.5,
                max_slope=30,
                avoid_overlap=False,
                clustering=True,
                density_per_area=0.5
            ),
            "props": PlacementRule(
                category="props",
                surface_types=["ground", "floor", "platform"],
                min_distance=0.5,
                max_slope=20,
                avoid_overlap=True,
                clustering=False,
                density_per_area=0.3
            ),
            "buildings": PlacementRule(
                category="buildings",
                surface_types=["ground", "foundation"],
                min_distance=5.0,
                max_slope=5,
                avoid_overlap=True,
                clustering=False,
                density_per_area=0.05
            )
        }
        
        # Style-specific asset preferences
        self.style_preferences = {
            "cyberpunk": {
                "preferred_categories": ["vehicles", "neon_signs", "tech_props"],
                "asset_keywords": ["neon", "cyber", "tech", "holographic", "futuristic"],
                "density_multiplier": 1.3
            },
            "medieval": {
                "preferred_categories": ["props", "vegetation", "buildings"],
                "asset_keywords": ["medieval", "stone", "wood", "ancient", "castle"],
                "density_multiplier": 0.8
            },
            "post-apocalyptic": {
                "preferred_categories": ["debris", "ruins", "abandoned_vehicles"],
                "asset_keywords": ["destroyed", "rusted", "abandoned", "broken"],
                "density_multiplier": 0.6
            },
            "natural": {
                "preferred_categories": ["vegetation", "rocks", "natural_props"],
                "asset_keywords": ["tree", "rock", "bush", "natural", "organic"],
                "density_multiplier": 1.5
            }
        }
        
        # Mock asset libraries (in production, this would be loaded from actual asset libraries)
        self.asset_libraries = {
            "default": AssetLibrary(
                name="default",
                path="./assets/default",
                categories={
                    "vehicles": ["car_sedan", "car_suv", "truck", "motorcycle"],
                    "street_furniture": ["bench", "lamp_post", "trash_can", "bus_stop"],
                    "vegetation": ["tree_oak", "tree_pine", "bush", "grass_patch"],
                    "props": ["barrel", "crate", "sign", "fence"]
                },
                style_tags=["realistic", "modern"]
            ),
            "cyberpunk": AssetLibrary(
                name="cyberpunk",
                path="./assets/cyberpunk",
                categories={
                    "vehicles": ["hover_car", "cyber_bike", "drone"],
                    "neon_signs": ["neon_sign_1", "hologram", "cyber_billboard"],
                    "tech_props": ["terminal", "antenna", "power_core"]
                },
                style_tags=["cyberpunk", "futuristic", "neon"]
            ),
            "medieval": AssetLibrary(
                name="medieval",
                path="./assets/medieval",
                categories={
                    "buildings": ["cottage", "tower", "wall_section"],
                    "props": ["barrel_wood", "cart", "weapon_rack", "campfire"],
                    "vegetation": ["oak_tree", "pine_tree", "herb_bush"]
                },
                style_tags=["medieval", "fantasy", "historical"]
            )
        }
        
        self.placement_prompt = """
        Plan asset placement for:
        Elements to place: {elements}
        Environment type: {environment_type}
        Style: {style}
        Scene boundaries: {boundaries}
        Available asset categories: {categories}
        
        Consider:
        1. Contextual appropriateness of each asset
        2. Realistic spacing and positioning
        3. Style consistency
        4. Natural clustering where appropriate
        5. Accessibility and logical placement
        
        Create a placement plan that feels natural and supports the scene's narrative.
        
        Respond with a JSON object matching the PlacementPlan schema.
        """
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process asset placement task"""
        task_params = input_data.get("parameters", {})
        elements = task_params.get("elements", [])
        environment_type = task_params.get("environment_type", "city")
        style = task_params.get("style", "realistic")
        
        # Get scene info from shared state
        shared_state = input_data.get("shared_state", {})
        scene_data = shared_state.get("metadata", {})
        
        self.logger.info(f"Placing assets for {style} {environment_type}: {elements}")
        
        # Generate placement plan
        placement_plan = await self._generate_placement_plan(
            elements, environment_type, style, scene_data
        )
        
        # Generate Blender commands
        blender_commands = self._generate_blender_commands(placement_plan)
        
        # Prepare output
        result = {
            "status": "success",
            "agent": self.config.name,
            "placement_plan": placement_plan.model_dump(),
            "blender_commands": blender_commands,
            "statistics": {
                "total_assets": len(placement_plan.assets),
                "categories_used": list(set(asset.category for asset in placement_plan.assets)),
                "libraries_used": placement_plan.libraries
            }
        }
        
        self.update_metadata("placement_plan", placement_plan.model_dump())
        
        return result
        
    async def _generate_placement_plan(
        self,
        elements: List[str],
        environment_type: str,
        style: str,
        scene_data: Dict[str, Any]
    ) -> PlacementPlan:
        """Generate intelligent asset placement plan"""
        
        # Determine scene boundaries (fallback if not provided)
        boundaries = scene_data.get("boundaries", (-50, -50, 50, 50))
        
        # Select appropriate asset libraries
        selected_libraries = self._select_libraries(style)
        
        # Get available categories
        available_categories = self._get_available_categories(selected_libraries)
        
        # Map elements to asset categories
        element_mapping = self._map_elements_to_categories(elements, available_categories)
        
        # Generate placement plan
        placement_plan = PlacementPlan(
            libraries=list(selected_libraries.keys())
        )
        
        # Place each element type
        for element, category in element_mapping.items():
            assets = await self._place_element_category(
                element, category, environment_type, style, boundaries, selected_libraries
            )
            placement_plan.assets.extend(assets)
            
        # Add contextual assets based on environment and style
        contextual_assets = self._add_contextual_assets(
            environment_type, style, boundaries, selected_libraries
        )
        placement_plan.assets.extend(contextual_assets)
        
        # Apply placement rules and collision detection
        placement_plan.assets = self._apply_placement_rules(placement_plan.assets)
        
        # Generate statistics
        placement_plan.statistics = self._calculate_statistics(placement_plan.assets)
        
        # Set rules used
        placement_plan.rules = [
            rule for rule in self.default_placement_rules.values()
            if rule.category in [asset.category for asset in placement_plan.assets]
        ]
        
        return placement_plan
        
    def _select_libraries(self, style: str) -> Dict[str, AssetLibrary]:
        """Select appropriate asset libraries for the style"""
        selected = {"default": self.asset_libraries["default"]}  # Always include default
        
        if style in self.asset_libraries:
            selected[style] = self.asset_libraries[style]
            
        return selected
        
    def _get_available_categories(self, libraries: Dict[str, AssetLibrary]) -> List[str]:
        """Get all available asset categories from selected libraries"""
        categories = set()
        for library in libraries.values():
            categories.update(library.categories.keys())
        return list(categories)
        
    def _map_elements_to_categories(
        self, 
        elements: List[str], 
        available_categories: List[str]
    ) -> Dict[str, str]:
        """Map requested elements to available asset categories"""
        mapping = {}
        
        # Define keyword-to-category mappings
        category_keywords = {
            "vehicles": ["car", "truck", "vehicle", "bike", "motorcycle"],
            "street_furniture": ["bench", "lamp", "sign", "furniture"],
            "vegetation": ["tree", "plant", "bush", "grass", "flower"],
            "props": ["barrel", "crate", "box", "object", "item"],
            "buildings": ["building", "house", "structure", "tower"]
        }
        
        for element in elements:
            element_lower = element.lower()
            matched = False
            
            # Try to match with available categories
            for category in available_categories:
                if category in element_lower or element_lower in category:
                    mapping[element] = category
                    matched = True
                    break
                    
            # Try keyword matching
            if not matched:
                for category, keywords in category_keywords.items():
                    if category in available_categories:
                        for keyword in keywords:
                            if keyword in element_lower:
                                mapping[element] = category
                                matched = True
                                break
                        if matched:
                            break
                            
            # Default fallback
            if not matched and available_categories:
                mapping[element] = "props"  # Default category
                
        return mapping
        
    async def _place_element_category(
        self,
        element: str,
        category: str,
        environment_type: str,
        style: str,
        boundaries: Tuple[float, float, float, float],
        libraries: Dict[str, AssetLibrary]
    ) -> List[AssetInstance]:
        """Place assets for a specific element category"""
        assets = []
        
        # Get placement rule for this category
        rule = self.default_placement_rules.get(category, PlacementRule(category=category))
        
        # Determine number of assets to place
        area = (boundaries[2] - boundaries[0]) * (boundaries[3] - boundaries[1])
        base_count = int(area * rule.density_per_area / 100)  # Scale down for demo
        
        # Apply style multiplier
        style_pref = self.style_preferences.get(style, {})
        multiplier = style_pref.get("density_multiplier", 1.0)
        count = max(1, int(base_count * multiplier))
        
        # Get available assets for this category
        available_assets = []
        for library in libraries.values():
            if category in library.categories:
                available_assets.extend(library.categories[category])
                
        if not available_assets:
            self.logger.warning(f"No assets available for category: {category}")
            return assets
            
        # Generate placements
        placed_positions = []
        
        for i in range(count):
            # Select asset
            asset_name = random.choice(available_assets)
            
            # Generate position
            position = self._generate_position(
                boundaries, rule, placed_positions, environment_type
            )
            
            if position is None:
                continue  # Skip if no valid position found
                
            # Generate rotation (random Y rotation for most objects)
            rotation = (0, random.uniform(0, 360), 0)
            
            # Generate scale variation
            scale_var = random.uniform(0.8, 1.2)
            scale = (scale_var, scale_var, scale_var)
            
            asset = AssetInstance(
                asset_name=asset_name,
                category=category,
                position=position,
                rotation=rotation,
                scale=scale,
                parameters={"element": element}
            )
            
            assets.append(asset)
            placed_positions.append(position)
            
        return assets
        
    def _generate_position(
        self,
        boundaries: Tuple[float, float, float, float],
        rule: PlacementRule,
        existing_positions: List[Tuple[float, float, float]],
        environment_type: str
    ) -> Optional[Tuple[float, float, float]]:
        """Generate a valid position for an asset"""
        min_x, min_y, max_x, max_y = boundaries
        
        max_attempts = 50
        for _ in range(max_attempts):
            # Generate random position
            x = random.uniform(min_x + 5, max_x - 5)  # Leave some margin
            y = random.uniform(min_y + 5, max_y - 5)
            z = 0  # Will be adjusted based on terrain
            
            position = (x, y, z)
            
            # Check minimum distance constraint
            if rule.min_distance > 0:
                too_close = False
                for existing_pos in existing_positions:
                    distance = ((x - existing_pos[0])**2 + (y - existing_pos[1])**2)**0.5
                    if distance < rule.min_distance:
                        too_close = True
                        break
                        
                if too_close:
                    continue
                    
            return position
            
        return None  # Failed to find valid position
        
    def _add_contextual_assets(
        self,
        environment_type: str,
        style: str,
        boundaries: Tuple[float, float, float, float],
        libraries: Dict[str, AssetLibrary]
    ) -> List[AssetInstance]:
        """Add contextual assets based on environment and style"""
        contextual_assets = []
        
        # Environment-specific additions
        if environment_type == "city":
            # Add some street furniture
            street_items = ["lamp_post", "bench", "trash_can"]
            for item in street_items[:2]:  # Limit for demo
                position = (
                    random.uniform(boundaries[0], boundaries[2]), 
                    random.uniform(boundaries[1], boundaries[3]), 
                    0
                )
                contextual_assets.append(AssetInstance(
                    asset_name=item,
                    category="street_furniture",
                    position=position,
                    rotation=(0, random.uniform(0, 360), 0),
                    parameters={"contextual": True}
                ))
                
        elif environment_type == "forest":
            # Add vegetation
            for _ in range(5):
                position = (
                    random.uniform(boundaries[0], boundaries[2]), 
                    random.uniform(boundaries[1], boundaries[3]), 
                    0
                )
                contextual_assets.append(AssetInstance(
                    asset_name="tree_oak",
                    category="vegetation",
                    position=position,
                    rotation=(0, random.uniform(0, 360), 0),
                    parameters={"contextual": True}
                ))
                
        return contextual_assets
        
    def _apply_placement_rules(self, assets: List[AssetInstance]) -> List[AssetInstance]:
        """Apply placement rules and resolve conflicts"""
        # Simple collision detection and resolution
        resolved_assets = []
        
        for asset in assets:
            rule = self.default_placement_rules.get(asset.category)
            if not rule:
                resolved_assets.append(asset)
                continue
                
            # Check for overlaps if required
            if rule.avoid_overlap:
                overlapping = False
                for existing in resolved_assets:
                    distance = (
                        (asset.position[0] - existing.position[0])**2 + 
                        (asset.position[1] - existing.position[1])**2
                    )**0.5
                    
                    min_dist = max(rule.min_distance, 1.0)
                    if distance < min_dist:
                        overlapping = True
                        break
                        
                if not overlapping:
                    resolved_assets.append(asset)
            else:
                resolved_assets.append(asset)
                
        return resolved_assets
        
    def _calculate_statistics(self, assets: List[AssetInstance]) -> Dict[str, int]:
        """Calculate placement statistics"""
        stats = {"total": len(assets)}
        
        # Count by category
        category_counts = {}
        for asset in assets:
            category_counts[asset.category] = category_counts.get(asset.category, 0) + 1
            
        stats.update(category_counts)
        
        return stats
        
    def _generate_blender_commands(self, plan: PlacementPlan) -> List[Dict[str, Any]]:
        """Generate Blender Python API commands for asset placement"""
        commands = []
        
        # Load asset libraries
        for library_name in plan.libraries:
            commands.append({
                "type": "load_asset_library",
                "params": {
                    "library_name": library_name,
                    "library_path": self.asset_libraries[library_name].path
                }
            })
            
        # Place assets
        for asset in plan.assets:
            commands.append({
                "type": "place_asset",
                "params": {
                    "asset_name": asset.asset_name,
                    "category": asset.category,
                    "position": asset.position,
                    "rotation": asset.rotation,
                    "scale": asset.scale,
                    "variation": asset.variation,
                    "parameters": asset.parameters
                }
            })
            
        return commands
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return "parameters" in input_data and isinstance(input_data["parameters"], dict)