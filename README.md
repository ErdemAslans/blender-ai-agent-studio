# 🎬 Blender AI Agent Studio

> **Transform natural language into stunning 3D scenes through intelligent multi-agent coordination**

A revolutionary AI-powered platform that democratizes 3D content creation by converting simple text descriptions into professional-quality Blender scenes using sophisticated multi-agent AI orchestration.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Blender](https://img.shields.io/badge/blender-4.0+-orange.svg)](https://blender.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

## 🚀 Overview

This system leverages specialized AI agents to handle different aspects of 3D scene generation:

- **Scene Director Agent**: Orchestrates the entire scene generation process
- **Environment Builder Agent**: Creates spatial compositions and structures
- **Lighting Designer Agent**: Manages illumination and atmospheric mood
- **Asset Placement Agent**: Handles intelligent object positioning
- **Effects Coordinator Agent**: Manages particles and weather effects

## 📋 Features

- Natural language to 3D scene generation
- Modular agent architecture for scalability
- Template-based generation system
- Real-time preview capabilities
- Professional Blender file output
- Style interpretation (cyberpunk, medieval, etc.)
- Atmospheric and weather effects
- Quality control and validation

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blender-ai-agent-studio.git
cd blender-ai-agent-studio
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## 🎯 Quick Start

### Basic Usage

```python
from blender_ai_studio import BlenderAIStudio

# Initialize the studio
studio = BlenderAIStudio()

# Generate a scene from natural language
scene = studio.generate_scene(
    prompt="Create a cyberpunk city street at night with neon lights and rain",
    style="cyberpunk",
    quality="high"
)

# Export to Blender file
scene.export("output/cyberpunk_city.blend")
```

### Using the Streamlit Demo

```bash
streamlit run demo/app.py
```

## 🏗️ Architecture

The system uses a multi-agent architecture where each agent specializes in a specific domain:

```
User Prompt
    ↓
Scene Director Agent (Orchestrator)
    ↓
Parallel Processing:
├── Environment Builder Agent
├── Lighting Designer Agent
├── Asset Placement Agent
└── Effects Coordinator Agent
    ↓
Quality Control & Validation
    ↓
Blender Scene Output
```

## 📚 Agent Descriptions

### Scene Director Agent
- Parses natural language prompts
- Breaks down scene requirements
- Coordinates other agents
- Ensures artistic coherence

### Environment Builder Agent
- Generates basic geometric structures
- Creates terrain and landscapes
- Establishes spatial composition
- Manages scene scale and proportions

### Lighting Designer Agent
- Configures lighting systems
- Sets material emission properties
- Adjusts color temperatures
- Creates atmospheric lighting

### Asset Placement Agent
- Positions objects intelligently
- Manages asset libraries
- Applies placement algorithms
- Ensures contextual appropriateness

### Effects Coordinator Agent
- Manages particle systems
- Creates weather effects
- Handles atmospheric elements
- Configures simulations

## 🎨 Supported Styles

- **Cyberpunk**: Neon lights, urban decay, high-tech elements
- **Fantasy Medieval**: Castle structures, mystical lighting, ancient atmosphere
- **Post-Apocalyptic**: Ruins, desolation, dramatic skies
- **Sci-Fi**: Futuristic designs, space elements, advanced technology
- **Natural**: Realistic landscapes, natural lighting, environmental details

## 🔧 Configuration

Edit `config/settings.yaml` to customize:

- Agent parameters
- Quality settings
- Template configurations
- Asset library paths
- Performance options

## 📝 Examples

Check the `examples/` directory for:

- Basic scene generation
- Style variations
- Complex compositions
- Animation setups
- Custom agent workflows

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Google's Agent Development Kit (ADK)
- Powered by Blender Python API
- Inspired by the creative 3D community

## 📧 Contact

For questions or support, please open an issue or contact us at support@blenderaistudio.com