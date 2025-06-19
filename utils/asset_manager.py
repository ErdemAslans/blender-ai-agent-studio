"""Asset management system for Blender AI Agent Studio"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
import time
import shutil
from urllib.parse import urlparse
import re

from utils.caching import get_cache, CacheType
from utils.logging_config import get_logger


class AssetType(str, Enum):
    """Types of 3D assets"""
    MODEL = "model"
    TEXTURE = "texture"
    MATERIAL = "material"
    HDRI = "hdri"
    SCENE_TEMPLATE = "scene_template"
    PRESET = "preset"


class AssetCategory(str, Enum):
    """Asset categories for organization"""
    ARCHITECTURE = "architecture"
    VEHICLES = "vehicles"
    CHARACTERS = "characters"
    PROPS = "props"
    NATURE = "nature"
    EFFECTS = "effects"
    LIGHTING = "lighting"
    MATERIALS = "materials"


@dataclass
class AssetMetadata:
    """Metadata for a 3D asset"""
    name: str
    asset_type: AssetType
    category: AssetCategory
    file_path: str
    style_tags: List[str]
    size_bytes: int
    polygon_count: Optional[int] = None
    texture_resolution: Optional[Tuple[int, int]] = None
    materials: List[str] = None
    description: str = ""
    author: str = ""
    license: str = "proprietary"
    version: str = "1.0"
    created_at: float = 0.0
    modified_at: float = 0.0
    usage_count: int = 0
    quality_score: float = 1.0
    performance_score: float = 1.0
    compatible_styles: List[str] = None
    
    def __post_init__(self):
        if self.materials is None:
            self.materials = []
        if self.compatible_styles is None:
            self.compatible_styles = []
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.modified_at == 0.0:
            self.modified_at = time.time()


@dataclass
class AssetLibrary:
    """Asset library configuration"""
    name: str
    path: str
    categories: List[AssetCategory]
    enabled: bool = True
    priority: int = 1
    cache_enabled: bool = True
    auto_scan: bool = True
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                # 3D model formats
                '.blend', '.obj', '.fbx', '.dae', '.gltf', '.glb', '.ply', '.stl',
                # Texture formats
                '.jpg', '.jpeg', '.png', '.tiff', '.tga', '.exr', '.hdr',
                # Material formats
                '.json', '.yaml', '.mtl'
            ]


class AssetScanner:
    """Scans directories for 3D assets and extracts metadata"""
    
    def __init__(self):
        self.logger = get_logger("asset_scanner")
        self.supported_formats = {
            # 3D Models
            '.blend': AssetType.MODEL,
            '.obj': AssetType.MODEL,
            '.fbx': AssetType.MODEL,
            '.dae': AssetType.MODEL,
            '.gltf': AssetType.MODEL,
            '.glb': AssetType.MODEL,
            '.ply': AssetType.MODEL,
            '.stl': AssetType.MODEL,
            
            # Textures
            '.jpg': AssetType.TEXTURE,
            '.jpeg': AssetType.TEXTURE,
            '.png': AssetType.TEXTURE,
            '.tiff': AssetType.TEXTURE,
            '.tga': AssetType.TEXTURE,
            '.exr': AssetType.TEXTURE,
            '.hdr': AssetType.HDRI,
            
            # Materials
            '.json': AssetType.MATERIAL,
            '.yaml': AssetType.MATERIAL,
            '.mtl': AssetType.MATERIAL,
        }
    
    def scan_directory(self, directory: str, library: AssetLibrary) -> List[AssetMetadata]:
        """Scan directory for assets and extract metadata"""
        assets = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            self.logger.warning(f"Asset directory does not exist: {directory}")
            return assets
        
        self.logger.info(f"Scanning asset directory: {directory}")
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    metadata = self._extract_metadata(file_path, library)
                    if metadata:
                        assets.append(metadata)
                except Exception as e:
                    self.logger.error(f"Error processing asset {file_path}: {e}")
        
        self.logger.info(f"Found {len(assets)} assets in {directory}")
        return assets
    
    def _extract_metadata(self, file_path: Path, library: AssetLibrary) -> Optional[AssetMetadata]:
        """Extract metadata from asset file"""
        try:
            # Basic file information
            stat = file_path.stat()
            file_extension = file_path.suffix.lower()
            asset_type = self.supported_formats.get(file_extension)
            
            if not asset_type:
                return None
            
            # Determine category from directory structure
            category = self._determine_category(file_path, library)
            
            # Extract style tags from filename and directory
            style_tags = self._extract_style_tags(file_path)
            
            # Basic metadata
            metadata = AssetMetadata(
                name=file_path.stem,
                asset_type=asset_type,
                category=category,
                file_path=str(file_path),
                style_tags=style_tags,
                size_bytes=stat.st_size,
                created_at=stat.st_ctime,
                modified_at=stat.st_mtime
            )
            
            # Try to extract detailed metadata based on file type
            if asset_type == AssetType.MODEL:
                self._extract_model_metadata(metadata)
            elif asset_type == AssetType.TEXTURE:
                self._extract_texture_metadata(metadata)
            elif asset_type == AssetType.MATERIAL:
                self._extract_material_metadata(metadata)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {e}")
            return None
    
    def _determine_category(self, file_path: Path, library: AssetLibrary) -> AssetCategory:
        """Determine asset category from path and filename"""
        path_str = str(file_path).lower()
        
        # Check directory names for category hints
        for category in AssetCategory:
            if category.value in path_str:
                return category
        
        # Check filename for category hints
        filename = file_path.name.lower()
        
        # Architecture keywords
        if any(keyword in filename for keyword in ['building', 'house', 'structure', 'wall', 'door', 'window']):
            return AssetCategory.ARCHITECTURE
        
        # Vehicle keywords
        if any(keyword in filename for keyword in ['car', 'truck', 'vehicle', 'bike', 'ship', 'plane']):
            return AssetCategory.VEHICLES
        
        # Character keywords
        if any(keyword in filename for keyword in ['character', 'person', 'human', 'npc', 'avatar']):
            return AssetCategory.CHARACTERS
        
        # Props keywords
        if any(keyword in filename for keyword in ['prop', 'object', 'furniture', 'item', 'tool']):
            return AssetCategory.PROPS
        
        # Nature keywords
        if any(keyword in filename for keyword in ['tree', 'plant', 'rock', 'terrain', 'landscape', 'nature']):
            return AssetCategory.NATURE
        
        # Default to props
        return AssetCategory.PROPS
    
    def _extract_style_tags(self, file_path: Path) -> List[str]:
        """Extract style tags from filename and directory path"""
        tags = []
        path_str = str(file_path).lower()
        
        # Style keywords to look for
        style_keywords = {
            'cyberpunk': ['cyber', 'punk', 'neon', 'futuristic', 'tech'],
            'medieval': ['medieval', 'castle', 'knight', 'fantasy', 'ancient'],
            'modern': ['modern', 'contemporary', 'current'],
            'scifi': ['scifi', 'sci-fi', 'space', 'alien', 'robot'],
            'horror': ['horror', 'scary', 'dark', 'creepy'],
            'cartoon': ['cartoon', 'toon', 'stylized', 'low-poly'],
            'realistic': ['realistic', 'real', 'photorealistic', 'pbr'],
            'industrial': ['industrial', 'factory', 'mechanical'],
            'nature': ['natural', 'organic', 'forest', 'outdoor'],
            'urban': ['urban', 'city', 'street', 'downtown'],
            'post_apocalyptic': ['apocalyptic', 'destroyed', 'ruined', 'wasteland']
        }
        
        for style, keywords in style_keywords.items():
            if any(keyword in path_str for keyword in keywords):
                tags.append(style)
        
        return tags
    
    def _extract_model_metadata(self, metadata: AssetMetadata):
        """Extract model-specific metadata (basic implementation)"""
        # This would typically require parsing the 3D file format
        # For now, we'll estimate based on file size
        file_size_mb = metadata.size_bytes / (1024 * 1024)
        
        # Rough estimates based on file size
        if file_size_mb < 1:
            metadata.polygon_count = 1000
            metadata.performance_score = 1.0
        elif file_size_mb < 10:
            metadata.polygon_count = 10000
            metadata.performance_score = 0.8
        elif file_size_mb < 50:
            metadata.polygon_count = 50000
            metadata.performance_score = 0.6
        else:
            metadata.polygon_count = 100000
            metadata.performance_score = 0.4
    
    def _extract_texture_metadata(self, metadata: AssetMetadata):
        """Extract texture-specific metadata"""
        try:
            from PIL import Image
            with Image.open(metadata.file_path) as img:
                metadata.texture_resolution = img.size
                
                # Quality score based on resolution
                total_pixels = img.size[0] * img.size[1]
                if total_pixels >= 4096 * 4096:
                    metadata.quality_score = 1.0
                elif total_pixels >= 2048 * 2048:
                    metadata.quality_score = 0.9
                elif total_pixels >= 1024 * 1024:
                    metadata.quality_score = 0.8
                else:
                    metadata.quality_score = 0.6
                    
        except ImportError:
            self.logger.warning("PIL not available for texture metadata extraction")
        except Exception as e:
            self.logger.warning(f"Could not extract texture metadata: {e}")
    
    def _extract_material_metadata(self, metadata: AssetMetadata):
        """Extract material-specific metadata"""
        try:
            if metadata.file_path.endswith('.json'):
                with open(metadata.file_path, 'r') as f:
                    material_data = json.load(f)
                    if isinstance(material_data, dict):
                        metadata.materials = [material_data.get('name', metadata.name)]
                        metadata.description = material_data.get('description', '')
                        
        except Exception as e:
            self.logger.warning(f"Could not extract material metadata: {e}")


class AssetManager:
    """Main asset management system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger("asset_manager")
        self.cache = get_cache()
        self.scanner = AssetScanner()
        
        # Asset storage
        self.libraries: List[AssetLibrary] = []
        self.assets: Dict[str, AssetMetadata] = {}
        self.asset_index: Dict[str, Set[str]] = {}  # category -> asset_ids
        
        # Load configuration
        self._load_configuration(config_path)
        
        # Initialize asset index
        self._initialize_index()
    
    def _load_configuration(self, config_path: Optional[str]):
        """Load asset library configuration"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self._parse_library_config(config.get('asset_libraries', []))
            except Exception as e:
                self.logger.error(f"Error loading asset configuration: {e}")
        
        # Add default libraries if none configured
        if not self.libraries:
            self._add_default_libraries()
    
    def _parse_library_config(self, library_configs: List[Dict[str, Any]]):
        """Parse library configuration from YAML"""
        for config in library_configs:
            try:
                categories = [AssetCategory(cat) for cat in config.get('categories', [])]
                library = AssetLibrary(
                    name=config['name'],
                    path=config['path'],
                    categories=categories,
                    enabled=config.get('enabled', True),
                    priority=config.get('priority', 1)
                )
                self.libraries.append(library)
            except Exception as e:
                self.logger.error(f"Error parsing library config: {e}")
    
    def _add_default_libraries(self):
        """Add default asset libraries"""
        default_libraries = [
            AssetLibrary(
                name="default",
                path="./assets/default",
                categories=[AssetCategory.ARCHITECTURE, AssetCategory.PROPS, AssetCategory.NATURE]
            ),
            AssetLibrary(
                name="cyberpunk",
                path="./assets/cyberpunk",
                categories=[AssetCategory.ARCHITECTURE, AssetCategory.VEHICLES, AssetCategory.PROPS]
            ),
            AssetLibrary(
                name="medieval",
                path="./assets/medieval",
                categories=[AssetCategory.ARCHITECTURE, AssetCategory.PROPS]
            ),
            AssetLibrary(
                name="nature",
                path="./assets/nature",
                categories=[AssetCategory.NATURE, AssetCategory.MATERIALS]
            )
        ]
        
        self.libraries.extend(default_libraries)
    
    def _initialize_index(self):
        """Initialize asset index for fast searching"""
        for category in AssetCategory:
            self.asset_index[category.value] = set()
    
    def scan_all_libraries(self, force_rescan: bool = False) -> Dict[str, int]:
        """Scan all enabled libraries for assets"""
        scan_results = {}
        
        for library in self.libraries:
            if not library.enabled:
                continue
                
            scan_count = self.scan_library(library, force_rescan)
            scan_results[library.name] = scan_count
        
        self.logger.info(f"Asset scan completed. Total assets: {len(self.assets)}")
        return scan_results
    
    def scan_library(self, library: AssetLibrary, force_rescan: bool = False) -> int:
        """Scan specific library for assets"""
        # Check cache first
        cache_key = f"library_scan:{library.name}:{os.path.getmtime(library.path) if os.path.exists(library.path) else 0}"
        
        if not force_rescan and library.cache_enabled:
            cached_assets = self.cache.get(cache_key)
            if cached_assets:
                self._add_assets_to_index(cached_assets)
                return len(cached_assets)
        
        # Scan directory
        assets = self.scanner.scan_directory(library.path, library)
        
        # Add to index
        self._add_assets_to_index(assets)
        
        # Cache results
        if library.cache_enabled:
            self.cache.put(
                cache_key,
                assets,
                CacheType.ASSET_METADATA,
                ttl_seconds=24 * 3600  # 24 hours
            )
        
        return len(assets)
    
    def _add_assets_to_index(self, assets: List[AssetMetadata]):
        """Add assets to internal index"""
        for asset in assets:
            asset_id = self._generate_asset_id(asset)
            self.assets[asset_id] = asset
            
            # Add to category index
            category_key = asset.category.value
            if category_key in self.asset_index:
                self.asset_index[category_key].add(asset_id)
    
    def _generate_asset_id(self, asset: AssetMetadata) -> str:
        """Generate unique asset ID"""
        content = f"{asset.file_path}:{asset.modified_at}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def search_assets(
        self,
        query: str = "",
        category: Optional[AssetCategory] = None,
        style_tags: Optional[List[str]] = None,
        asset_type: Optional[AssetType] = None,
        max_results: int = 50
    ) -> List[AssetMetadata]:
        """Search for assets matching criteria"""
        
        results = []
        search_pool = set()
        
        # Filter by category first
        if category:
            search_pool = self.asset_index.get(category.value, set())
        else:
            # Search all assets
            for asset_ids in self.asset_index.values():
                search_pool.update(asset_ids)
        
        # Apply filters
        for asset_id in search_pool:
            asset = self.assets.get(asset_id)
            if not asset:
                continue
            
            # Filter by asset type
            if asset_type and asset.asset_type != asset_type:
                continue
            
            # Filter by style tags
            if style_tags:
                if not any(tag in asset.style_tags for tag in style_tags):
                    continue
            
            # Filter by query (simple text matching)
            if query:
                searchable_text = f"{asset.name} {asset.description} {' '.join(asset.style_tags)}".lower()
                if query.lower() not in searchable_text:
                    continue
            
            results.append(asset)
            
            if len(results) >= max_results:
                break
        
        # Sort by relevance (usage count, quality score)
        results.sort(key=lambda a: (a.usage_count, a.quality_score), reverse=True)
        
        return results
    
    def get_asset_by_id(self, asset_id: str) -> Optional[AssetMetadata]:
        """Get asset by ID"""
        return self.assets.get(asset_id)
    
    def get_assets_by_category(self, category: AssetCategory) -> List[AssetMetadata]:
        """Get all assets in a category"""
        asset_ids = self.asset_index.get(category.value, set())
        return [self.assets[asset_id] for asset_id in asset_ids if asset_id in self.assets]
    
    def get_compatible_assets(
        self,
        style: str,
        category: Optional[AssetCategory] = None,
        max_results: int = 20
    ) -> List[AssetMetadata]:
        """Get assets compatible with a specific style"""
        style_tags = [style.lower()]
        return self.search_assets(
            style_tags=style_tags,
            category=category,
            max_results=max_results
        )
    
    def record_asset_usage(self, asset_id: str):
        """Record asset usage for popularity tracking"""
        if asset_id in self.assets:
            self.assets[asset_id].usage_count += 1
    
    def get_popular_assets(self, category: Optional[AssetCategory] = None, limit: int = 10) -> List[AssetMetadata]:
        """Get most popular assets"""
        if category:
            candidates = self.get_assets_by_category(category)
        else:
            candidates = list(self.assets.values())
        
        # Sort by usage count
        candidates.sort(key=lambda a: a.usage_count, reverse=True)
        return candidates[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get asset management statistics"""
        stats = {
            "total_assets": len(self.assets),
            "libraries": len(self.libraries),
            "categories": {}
        }
        
        # Category breakdown
        for category in AssetCategory:
            asset_count = len(self.asset_index.get(category.value, set()))
            if asset_count > 0:
                stats["categories"][category.value] = asset_count
        
        # Type breakdown
        type_counts = {}
        for asset in self.assets.values():
            asset_type = asset.asset_type.value
            type_counts[asset_type] = type_counts.get(asset_type, 0) + 1
        
        stats["asset_types"] = type_counts
        
        return stats
    
    def create_asset_preset(
        self,
        name: str,
        asset_ids: List[str],
        description: str = "",
        style: str = ""
    ) -> str:
        """Create a preset combining multiple assets"""
        preset_data = {
            "name": name,
            "description": description,
            "style": style,
            "assets": asset_ids,
            "created_at": time.time()
        }
        
        preset_id = hashlib.md5(f"{name}:{time.time()}".encode()).hexdigest()[:16]
        
        # Cache the preset
        self.cache.put(
            f"asset_preset:{preset_id}",
            preset_data,
            CacheType.SCENE_TEMPLATE,
            ttl_seconds=30 * 24 * 3600  # 30 days
        )
        
        return preset_id
    
    def get_asset_preset(self, preset_id: str) -> Optional[Dict[str, Any]]:
        """Get asset preset by ID"""
        return self.cache.get(f"asset_preset:{preset_id}")


# Global asset manager instance
asset_manager = AssetManager()


def get_asset_manager() -> AssetManager:
    """Get the global asset manager instance"""
    return asset_manager