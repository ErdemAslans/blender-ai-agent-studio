"""Configuration management for Blender AI Agent Studio"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml


class AgentModelConfig(BaseModel):
    """Configuration for individual agent AI models"""
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3


class PerformanceConfig(BaseModel):
    """Performance and resource configuration"""
    max_concurrent_operations: int = 5
    memory_limit_mb: int = 8192
    cache_size_mb: int = 1024
    enable_caching: bool = True
    enable_profiling: bool = False


class BlenderConfig(BaseModel):
    """Blender-specific configuration"""
    blender_path: Optional[str] = None
    timeout: int = 300
    use_gpu: bool = True
    auto_cleanup: bool = True
    max_output_size_mb: int = 500


class QualityPreset(BaseModel):
    """Quality preset configuration"""
    samples: int
    resolution_scale: float
    simplify_subdivision: int


class OutputConfig(BaseModel):
    """Output and rendering configuration"""
    output_directory: str = "./output"
    default_quality: str = "high"
    max_output_size_mb: int = 500
    auto_cleanup: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    to_file: bool = True
    file_path: str = "./logs/blender_ai_studio.log"


class WebConfig(BaseModel):
    """Web interface configuration"""
    enable: bool = True
    host: str = "localhost"
    port: int = 8080
    enable_auth: bool = False
    session_secret: str = "change-this-secret-key"
    enable_cors: bool = True
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]


class SecurityConfig(BaseModel):
    """Security configuration"""
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 10
    max_scene_complexity: int = 1000


class Settings(BaseSettings):
    """Main application settings"""
    
    # AI Configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Feature Flags
    development_mode: bool = Field(False, env="DEVELOPMENT_MODE")
    test_mode: bool = Field(False, env="TEST_MODE")
    enable_experimental: bool = Field(False, env="ENABLE_EXPERIMENTAL")
    enable_collaboration: bool = Field(False, env="ENABLE_COLLABORATION")
    enable_cloud_storage: bool = Field(False, env="ENABLE_CLOUD_STORAGE")
    
    # Asset Configuration
    asset_library_paths: str = Field("./assets/default", env="ASSET_LIBRARY_PATHS")
    enable_online_assets: bool = Field(False, env="ENABLE_ONLINE_ASSETS")
    asset_cache_hours: int = Field(24, env="ASSET_CACHE_HOURS")
    
    # Database
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    enable_scene_history: bool = Field(True, env="ENABLE_SCENE_HISTORY")
    
    # Cloud Storage
    cloud_storage_bucket: Optional[str] = Field(None, env="CLOUD_STORAGE_BUCKET")
    cloud_storage_region: str = Field("us-east-1", env="CLOUD_STORAGE_REGION")
    
    # Redis for distributed processing
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    enable_distributed: bool = Field(False, env="ENABLE_DISTRIBUTED")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "./config/settings.yaml"
        self.settings = Settings()
        self._yaml_config = self._load_yaml_config()
        self._agent_configs = self._load_agent_configs()
        self._quality_presets = self._load_quality_presets()
        
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            print(f"Warning: Could not load YAML config: {e}")
            return {}
    
    def _load_agent_configs(self) -> Dict[str, AgentModelConfig]:
        """Load agent-specific configurations"""
        agents_config = self._yaml_config.get("agents", {})
        configs = {}
        
        default_config = {
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.7,
            "max_tokens": 4096,
            "timeout": 60,
            "max_retries": 3
        }
        
        agent_names = [
            "scene_director", "environment_builder", "lighting_designer",
            "asset_placement", "effects_coordinator"
        ]
        
        for agent_name in agent_names:
            config_dict = {**default_config, **agents_config.get(agent_name, {})}
            configs[agent_name] = AgentModelConfig(**config_dict)
            
        return configs
    
    def _load_quality_presets(self) -> Dict[str, QualityPreset]:
        """Load quality presets"""
        presets_config = self._yaml_config.get("quality_presets", {})
        
        defaults = {
            "draft": {"samples": 32, "resolution_scale": 0.5, "simplify_subdivision": 2},
            "preview": {"samples": 64, "resolution_scale": 0.75, "simplify_subdivision": 1},
            "high": {"samples": 128, "resolution_scale": 1.0, "simplify_subdivision": 0},
            "ultra": {"samples": 256, "resolution_scale": 1.0, "simplify_subdivision": 0}
        }
        
        presets = {}
        for name, config in defaults.items():
            preset_config = {**config, **presets_config.get(name, {})}
            presets[name] = QualityPreset(**preset_config)
            
        return presets
    
    @property
    def performance(self) -> PerformanceConfig:
        """Get performance configuration"""
        return PerformanceConfig(
            max_concurrent_operations=int(os.getenv("MAX_CONCURRENT_OPERATIONS", "5")),
            memory_limit_mb=int(os.getenv("MEMORY_LIMIT_MB", "8192")),
            cache_size_mb=int(os.getenv("CACHE_SIZE_MB", "1024")),
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            enable_profiling=os.getenv("ENABLE_PROFILING", "false").lower() == "true"
        )
    
    @property
    def blender(self) -> BlenderConfig:
        """Get Blender configuration"""
        return BlenderConfig(
            blender_path=os.getenv("BLENDER_PATH"),
            timeout=int(os.getenv("BLENDER_TIMEOUT", "300")),
            use_gpu=os.getenv("BLENDER_USE_GPU", "true").lower() == "true",
            auto_cleanup=os.getenv("AUTO_CLEANUP", "true").lower() == "true",
            max_output_size_mb=int(os.getenv("MAX_OUTPUT_SIZE_MB", "500"))
        )
    
    @property
    def output(self) -> OutputConfig:
        """Get output configuration"""
        return OutputConfig(
            output_directory=os.getenv("OUTPUT_DIRECTORY", "./output"),
            default_quality=os.getenv("DEFAULT_QUALITY", "high"),
            max_output_size_mb=int(os.getenv("MAX_OUTPUT_SIZE_MB", "500")),
            auto_cleanup=os.getenv("AUTO_CLEANUP", "true").lower() == "true"
        )
    
    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration"""
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "json"),
            to_file=os.getenv("LOG_TO_FILE", "true").lower() == "true",
            file_path=os.getenv("LOG_FILE_PATH", "./logs/blender_ai_studio.log")
        )
    
    @property
    def web(self) -> WebConfig:
        """Get web interface configuration"""
        cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
        return WebConfig(
            enable=os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true",
            host=os.getenv("WEB_HOST", "localhost"),
            port=int(os.getenv("WEB_PORT", "8080")),
            enable_auth=os.getenv("ENABLE_AUTH", "false").lower() == "true",
            session_secret=os.getenv("SESSION_SECRET", "change-this-secret-key"),
            enable_cors=os.getenv("ENABLE_CORS", "true").lower() == "true",
            cors_origins=[origin.strip() for origin in cors_origins.split(",")]
        )
    
    @property
    def security(self) -> SecurityConfig:
        """Get security configuration"""
        return SecurityConfig(
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")),
            max_scene_complexity=int(os.getenv("MAX_SCENE_COMPLEXITY", "1000"))
        )
    
    def get_agent_config(self, agent_name: str) -> AgentModelConfig:
        """Get configuration for specific agent"""
        return self._agent_configs.get(agent_name, AgentModelConfig())
    
    def get_quality_preset(self, preset_name: str) -> QualityPreset:
        """Get quality preset configuration"""
        return self._quality_presets.get(preset_name, self._quality_presets["high"])
    
    def get_asset_library_paths(self) -> List[str]:
        """Get list of asset library paths"""
        paths_str = self.settings.asset_library_paths
        return [path.strip() for path in paths_str.split(",") if path.strip()]
    
    def get_template_config(self, template_type: str, template_name: str) -> Dict[str, Any]:
        """Get template configuration (styles, etc.)"""
        templates = self._yaml_config.get("templates", {})
        return templates.get(template_type, {}).get(template_name, {})
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required API keys
        if not self.settings.google_api_key or self.settings.google_api_key == "your_gemini_api_key_here":
            issues.append("GOOGLE_API_KEY is required and must be set to a valid API key")
        
        # Check output directory
        output_dir = Path(self.output.output_directory)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory {output_dir}: {e}")
        
        # Check asset library paths
        for path in self.get_asset_library_paths():
            if not os.path.exists(path):
                issues.append(f"Asset library path does not exist: {path}")
        
        # Check log directory
        log_file = Path(self.logging.file_path)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create log directory {log_file.parent}: {e}")
        
        return issues
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.output.output_directory,
            Path(self.logging.file_path).parent,
            "./cache",
            "./temp"
        ]
        
        for asset_path in self.get_asset_library_paths():
            directories.append(asset_path)
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config


def reload_config(config_path: Optional[str] = None):
    """Reload configuration from files"""
    global config
    config = ConfigManager(config_path)
    return config