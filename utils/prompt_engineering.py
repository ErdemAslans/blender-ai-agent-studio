"""Advanced prompt engineering and response validation for Blender AI Agent Studio"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
import hashlib
from dataclasses import dataclass


class PromptType(str, Enum):
    """Types of prompts for different agents"""
    SCENE_ANALYSIS = "scene_analysis"
    ENVIRONMENT_GENERATION = "environment_generation"
    LIGHTING_DESIGN = "lighting_design"
    ASSET_PLACEMENT = "asset_placement"
    EFFECTS_COORDINATION = "effects_coordination"
    QUALITY_REVIEW = "quality_review"


class ResponseFormat(str, Enum):
    """Expected response formats"""
    JSON = "json"
    STRUCTURED_TEXT = "structured_text"
    BLENDER_COMMANDS = "blender_commands"
    ANALYSIS = "analysis"


@dataclass
class PromptTemplate:
    """Template for prompt generation"""
    system_prompt: str
    user_prompt_template: str
    response_format: ResponseFormat
    validation_schema: Optional[Dict[str, Any]] = None
    examples: Optional[List[Dict[str, str]]] = None
    temperature: float = 0.7
    max_tokens: int = 4096


class PromptOptimizer:
    """Optimizes prompts based on success rates and performance"""
    
    def __init__(self):
        self.success_rates: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.optimization_cache: Dict[str, str] = {}
    
    def record_success(self, prompt_hash: str, success: bool, response_time: float):
        """Record the success/failure of a prompt"""
        if prompt_hash not in self.success_rates:
            self.success_rates[prompt_hash] = 0.0
            self.performance_history[prompt_hash] = []
        
        # Update success rate (exponential moving average)
        current_rate = self.success_rates[prompt_hash]
        self.success_rates[prompt_hash] = 0.8 * current_rate + 0.2 * (1.0 if success else 0.0)
        
        # Record response time
        self.performance_history[prompt_hash].append(response_time)
        if len(self.performance_history[prompt_hash]) > 100:
            self.performance_history[prompt_hash] = self.performance_history[prompt_hash][-100:]
    
    def get_optimization_suggestion(self, prompt: str) -> Optional[str]:
        """Get optimization suggestions for a prompt"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        if prompt_hash in self.success_rates:
            success_rate = self.success_rates[prompt_hash]
            if success_rate < 0.7:  # Poor success rate
                return self._generate_optimization(prompt, success_rate)
        
        return None
    
    def _generate_optimization(self, prompt: str, success_rate: float) -> str:
        """Generate optimization suggestions"""
        suggestions = []
        
        if success_rate < 0.3:
            suggestions.append("Consider simplifying the prompt and breaking it into smaller parts")
        elif success_rate < 0.5:
            suggestions.append("Try adding more specific examples or constraints")
        elif success_rate < 0.7:
            suggestions.append("Consider adjusting the temperature or adding more context")
        
        return "; ".join(suggestions)


class PromptEngineer:
    """Advanced prompt engineering for different agent types"""
    
    def __init__(self):
        self.optimizer = PromptOptimizer()
        self.templates = self._load_templates()
        self.context_cache: Dict[str, Any] = {}
    
    def _load_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Load prompt templates for different agent types"""
        return {
            PromptType.SCENE_ANALYSIS: PromptTemplate(
                system_prompt="""You are an expert 3D scene director with deep understanding of visual composition, spatial relationships, and artistic direction. Your role is to analyze natural language scene descriptions and extract structured requirements for 3D scene generation.

Key capabilities:
- Understand artistic styles (cyberpunk, medieval, realistic, fantasy, etc.)
- Interpret spatial relationships and environmental context
- Identify required elements, mood, and atmosphere
- Plan optimal execution order for multi-agent scene generation
- Ensure visual coherence and artistic consistency

Always respond with valid JSON that matches the expected schema exactly.""",
                
                user_prompt_template="""Analyze this scene description and create structured requirements:

SCENE DESCRIPTION: "{prompt}"
STYLE: {style}
QUALITY: {quality}

Extract and structure the following information:
1. Main elements and objects needed
2. Environmental context and setting
3. Mood, atmosphere, and artistic direction
4. Spatial relationships and composition
5. Technical requirements (lighting, effects, materials)
6. Execution plan for agent coordination

Respond with JSON containing:
- scene_type: (indoor/outdoor/mixed)
- complexity: (simple/moderate/complex/ultra)
- elements: [list of main elements]
- environment: {environment details}
- mood: {atmosphere and mood}
- style_requirements: {style-specific needs}
- agent_tasks: {tasks for each specialist agent}
- execution_order: [ordered phases of execution]

Focus on artistic vision and practical implementation requirements.""",
                
                response_format=ResponseFormat.JSON,
                temperature=0.8,
                max_tokens=2048
            ),
            
            PromptType.ENVIRONMENT_GENERATION: PromptTemplate(
                system_prompt="""You are a master environment artist specializing in procedural world generation for 3D scenes. You excel at creating compelling, believable environments that support the narrative and aesthetic goals of each scene.

Expertise areas:
- Terrain generation and landscape design
- Architectural placement and urban planning
- Natural environment simulation
- Style-appropriate environmental storytelling
- Performance-optimized geometry generation

Generate precise technical specifications for environment creation.""",
                
                user_prompt_template="""Create an environment based on these requirements:

SCENE REQUIREMENTS: {requirements}
STYLE: {style}
COMPLEXITY: {complexity}
ENVIRONMENT TYPE: {environment_type}

Design environmental elements including:
1. Terrain characteristics (topology, materials, scale)
2. Architectural structures (buildings, props, infrastructure)
3. Natural elements (vegetation, water, geological features)
4. Atmospheric conditions (weather, time of day effects)
5. Style-specific details that enhance the aesthetic

Consider:
- Spatial relationships and navigation flow
- Visual hierarchy and focal points
- Performance optimization (LOD, culling)
- Environmental storytelling elements

Provide specific technical parameters for:
- Terrain generation (size, subdivision, height maps)
- Structure placement (positions, rotations, scales)
- Material assignments and surface properties
- Environmental effects and atmospheric conditions""",
                
                response_format=ResponseFormat.STRUCTURED_TEXT,
                temperature=0.6,
                max_tokens=2048
            ),
            
            PromptType.LIGHTING_DESIGN: PromptTemplate(
                system_prompt="""You are a cinematographer and lighting director with expertise in 3D rendering and mood creation. You understand how lighting shapes emotion, guides attention, and creates atmosphere in virtual environments.

Specializations:
- Cinematic lighting principles (3-point lighting, key/fill/rim)
- Time-of-day and weather lighting simulation
- Style-specific lighting signatures (noir, cyberpunk, fantasy)
- Technical lighting optimization for real-time and offline rendering
- Color theory and mood psychology through illumination

Create lighting setups that serve both artistic and technical goals.""",
                
                user_prompt_template="""Design lighting for this scene:

SCENE CONTEXT: {scene_context}
MOOD: {mood}
STYLE: {style}
TIME OF DAY: {time_of_day}
WEATHER: {weather}

Create a lighting setup that:
1. Establishes the intended mood and atmosphere
2. Follows style-appropriate lighting conventions
3. Guides viewer attention to focal points
4. Supports the narrative and emotional tone
5. Maintains technical efficiency

Specify:
- Primary light sources (sun, artificial, practical)
- Light types, positions, and intensities
- Color temperatures and tinting
- Shadow characteristics and contrast ratios
- Atmospheric effects (fog, bloom, exposure)
- Special lighting effects for style enhancement

Balance artistic vision with rendering performance.""",
                
                response_format=ResponseFormat.STRUCTURED_TEXT,
                temperature=0.7,
                max_tokens=1536
            ),
            
            PromptType.ASSET_PLACEMENT: PromptTemplate(
                system_prompt="""You are a scene composition expert specializing in object placement, spatial design, and environmental storytelling through asset arrangement.

Core competencies:
- Spatial composition and visual balance
- Asset library management and selection
- Environmental storytelling through object placement
- Performance optimization through intelligent instancing
- Style-consistent asset integration

Create believable, engaging environments through strategic asset placement.""",
                
                user_prompt_template="""Place assets for this environment:

ENVIRONMENT: {environment_description}
AVAILABLE_ASSETS: {asset_categories}
STYLE: {style}
DENSITY: {density_preference}
FOCAL_POINTS: {focal_points}

Strategically place assets to:
1. Support environmental storytelling
2. Create visual interest and navigation cues
3. Maintain style consistency
4. Optimize performance through smart instancing
5. Establish proper scale relationships

Consider:
- Logical groupings and clustering
- Accessibility and navigation paths
- Visual hierarchy and emphasis
- Performance implications (draw calls, complexity)
- Environmental wear and contextual details

Specify placements with:
- Asset selection criteria
- Position, rotation, and scale variations
- Grouping strategies for related objects
- Performance optimization techniques""",
                
                response_format=ResponseFormat.STRUCTURED_TEXT,
                temperature=0.6,
                max_tokens=1536
            ),
            
            PromptType.EFFECTS_COORDINATION: PromptTemplate(
                system_prompt="""You are a visual effects supervisor with expertise in particle systems, atmospheric effects, and post-processing enhancement for 3D environments.

Technical expertise:
- Particle system design and optimization
- Weather and atmospheric simulation
- Post-processing and screen-space effects
- Performance profiling and optimization
- Style-appropriate effects integration

Create effects that enhance realism and atmosphere while maintaining performance.""",
                
                user_prompt_template="""Design effects for this scene:

SCENE_DESCRIPTION: {scene_description}
ATMOSPHERE: {atmosphere}
WEATHER: {weather_conditions}
STYLE: {style}
PERFORMANCE_TARGET: {performance_target}

Create effects that:
1. Enhance atmospheric immersion
2. Support the environmental storytelling
3. Match the artistic style requirements
4. Maintain target performance levels
5. Integrate seamlessly with other scene elements

Design effects for:
- Weather simulation (rain, snow, fog, wind)
- Atmospheric conditions (dust, smoke, steam)
- Environmental details (falling leaves, floating particles)
- Style-specific effects (neon glow, magical auras, industrial steam)
- Post-processing enhancements

Balance visual impact with computational efficiency.""",
                
                response_format=ResponseFormat.STRUCTURED_TEXT,
                temperature=0.7,
                max_tokens=1536
            ),
            
            PromptType.QUALITY_REVIEW: PromptTemplate(
                system_prompt="""You are a technical director and quality assurance specialist for 3D scene generation. You evaluate scenes for technical correctness, artistic coherence, and user satisfaction.

Evaluation criteria:
- Technical validity (geometry, materials, lighting)
- Artistic coherence and style consistency
- Performance optimization and efficiency
- User requirement fulfillment
- Production-ready quality standards

Provide constructive feedback for scene improvement.""",
                
                user_prompt_template="""Review this generated scene:

ORIGINAL_PROMPT: "{original_prompt}"
SCENE_REQUIREMENTS: {requirements}
GENERATION_RESULTS: {results}
PERFORMANCE_METRICS: {performance}

Evaluate the scene across these dimensions:
1. Requirement fulfillment - Does it match the user's intent?
2. Technical quality - Are there any technical issues?
3. Artistic coherence - Is the style consistent?
4. Performance optimization - Is it efficiently constructed?
5. User satisfaction potential - Will users be satisfied?

Provide:
- Overall quality score (1-10)
- Specific strengths and achievements
- Areas needing improvement
- Actionable recommendations for enhancement
- Technical issues requiring attention

Be constructive and specific in feedback.""",
                
                response_format=ResponseFormat.ANALYSIS,
                temperature=0.5,
                max_tokens=1024
            )
        }
    
    def generate_prompt(
        self,
        prompt_type: PromptType,
        context: Dict[str, Any],
        user_input: str = "",
        optimization_level: int = 1
    ) -> Tuple[str, str]:
        """Generate optimized prompt for specific agent type"""
        
        template = self.templates[prompt_type]
        
        # Build system prompt
        system_prompt = template.system_prompt
        
        # Build user prompt with context substitution
        user_prompt = template.user_prompt_template.format(
            prompt=user_input,
            **context
        )
        
        # Apply optimizations
        if optimization_level > 0:
            user_prompt = self._apply_optimizations(user_prompt, template, optimization_level)
        
        return system_prompt, user_prompt
    
    def _apply_optimizations(
        self,
        prompt: str,
        template: PromptTemplate,
        level: int
    ) -> str:
        """Apply prompt optimizations based on level"""
        
        if level >= 1:
            # Add clarity enhancements
            prompt = self._add_clarity_markers(prompt)
        
        if level >= 2:
            # Add examples if available
            if template.examples:
                prompt = self._add_examples(prompt, template.examples)
        
        if level >= 3:
            # Add constraint reinforcement
            prompt = self._add_constraints(prompt, template.response_format)
        
        return prompt
    
    def _add_clarity_markers(self, prompt: str) -> str:
        """Add clarity markers to improve response accuracy"""
        clarity_additions = [
            "\nIMPORTANT: Be specific and precise in your response.",
            "\nFOCUS: Address all requested aspects systematically.",
            "\nQUALITY: Ensure technical accuracy and artistic coherence."
        ]
        
        return prompt + "\n".join(clarity_additions)
    
    def _add_examples(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        """Add examples to guide response format"""
        if not examples:
            return prompt
        
        example_text = "\n\nEXAMPLES:\n"
        for i, example in enumerate(examples[:2], 1):  # Limit to 2 examples
            example_text += f"\nExample {i}:\n"
            example_text += f"Input: {example.get('input', '')}\n"
            example_text += f"Output: {example.get('output', '')}\n"
        
        return prompt + example_text
    
    def _add_constraints(self, prompt: str, response_format: ResponseFormat) -> str:
        """Add format constraints to ensure proper response structure"""
        format_constraints = {
            ResponseFormat.JSON: "\n\nCONSTRAINTS: Respond ONLY with valid JSON. No additional text or explanations.",
            ResponseFormat.STRUCTURED_TEXT: "\n\nCONSTRAINTS: Use clear section headers and structured formatting.",
            ResponseFormat.BLENDER_COMMANDS: "\n\nCONSTRAINTS: Generate only executable Blender Python commands.",
            ResponseFormat.ANALYSIS: "\n\nCONSTRAINTS: Provide structured analysis with specific scores and recommendations."
        }
        
        return prompt + format_constraints.get(response_format, "")
    
    def validate_response(
        self,
        response: str,
        expected_format: ResponseFormat,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate AI response against expected format and schema"""
        
        try:
            if expected_format == ResponseFormat.JSON:
                return self._validate_json_response(response, schema)
            elif expected_format == ResponseFormat.STRUCTURED_TEXT:
                return self._validate_structured_text(response)
            elif expected_format == ResponseFormat.BLENDER_COMMANDS:
                return self._validate_blender_commands(response)
            elif expected_format == ResponseFormat.ANALYSIS:
                return self._validate_analysis(response)
            else:
                return True, None, {"raw_response": response}
                
        except Exception as e:
            return False, f"Validation error: {str(e)}", None
    
    def _validate_json_response(
        self,
        response: str,
        schema: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate JSON response"""
        try:
            # Clean response (remove any surrounding text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = response.strip()
            
            parsed = json.loads(json_str)
            
            # Schema validation would go here if schema is provided
            if schema:
                # Placeholder for schema validation
                pass
            
            return True, None, parsed
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}", None
    
    def _validate_structured_text(self, response: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate structured text response"""
        # Check for basic structure (headers, organization)
        if len(response.strip()) < 50:
            return False, "Response too short for structured text", None
        
        # Look for section markers
        has_structure = bool(re.search(r'^\s*\d+\.|\*\s+|-\s+|#{1,6}\s+', response, re.MULTILINE))
        
        if not has_structure:
            return False, "Response lacks clear structure", None
        
        return True, None, {"structured_text": response}
    
    def _validate_blender_commands(self, response: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate Blender commands response"""
        # Check for Python/Blender specific patterns
        blender_patterns = [
            r'bpy\.',
            r'bmesh\.',
            r'mathutils\.',
            r'\.add\(',
            r'\.new\(',
            r'context\.'
        ]
        
        has_blender_commands = any(re.search(pattern, response) for pattern in blender_patterns)
        
        if not has_blender_commands:
            return False, "Response doesn't contain recognizable Blender commands", None
        
        return True, None, {"blender_commands": response}
    
    def _validate_analysis(self, response: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate analysis response"""
        # Look for analysis markers
        analysis_patterns = [
            r'score.*\d+',
            r'rating.*\d+',
            r'quality.*\d+',
            r'recommend',
            r'improve',
            r'strength',
            r'weakness'
        ]
        
        has_analysis = any(re.search(pattern, response, re.IGNORECASE) for pattern in analysis_patterns)
        
        if not has_analysis:
            return False, "Response doesn't contain recognizable analysis elements", None
        
        return True, None, {"analysis": response}
    
    def enhance_context(self, base_context: Dict[str, Any], agent_type: PromptType) -> Dict[str, Any]:
        """Enhance context with agent-specific information"""
        enhanced = base_context.copy()
        
        # Add agent-specific context enhancements
        if agent_type == PromptType.SCENE_ANALYSIS:
            enhanced.update({
                "analysis_depth": "comprehensive",
                "focus_areas": ["composition", "style", "technical_requirements"]
            })
        elif agent_type == PromptType.ENVIRONMENT_GENERATION:
            enhanced.update({
                "generation_mode": "procedural",
                "optimization_target": "balanced_quality_performance"
            })
        elif agent_type == PromptType.LIGHTING_DESIGN:
            enhanced.update({
                "lighting_approach": "cinematic",
                "mood_priority": "primary"
            })
        elif agent_type == PromptType.ASSET_PLACEMENT:
            enhanced.update({
                "placement_strategy": "contextual_storytelling",
                "performance_awareness": "high"
            })
        elif agent_type == PromptType.EFFECTS_COORDINATION:
            enhanced.update({
                "effects_priority": "atmospheric_enhancement",
                "performance_budget": "medium"
            })
        
        return enhanced
    
    def record_prompt_performance(
        self,
        prompt_type: PromptType,
        prompt: str,
        success: bool,
        response_time: float,
        quality_score: Optional[float] = None
    ):
        """Record prompt performance for optimization"""
        prompt_hash = hashlib.md5(f"{prompt_type}:{prompt}".encode()).hexdigest()
        self.optimizer.record_success(prompt_hash, success, response_time)
        
        if quality_score is not None:
            # Record quality metrics for future optimization
            if prompt_hash not in self.context_cache:
                self.context_cache[prompt_hash] = {}
            self.context_cache[prompt_hash]["quality_score"] = quality_score


# Global prompt engineer instance
prompt_engineer = PromptEngineer()


def get_prompt_engineer() -> PromptEngineer:
    """Get the global prompt engineer instance"""
    return prompt_engineer