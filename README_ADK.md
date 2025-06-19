# Blender AI Agent Studio - ADK Integration

🎬 AI-powered 3D scene generation using Google's Agent Development Kit (ADK) and Vertex AI.

## Quick Start with ADK Web

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Google API key and GCP project
   ```

3. **Start ADK Web UI:**
   ```bash
   adk web
   ```

4. **Access the interface:**
   Open http://localhost:8000 in your browser

## Available Agent Tools

### Scene Director Agent
The main orchestrator agent with the following tools:

- `analyze_scene_prompt(prompt, style, quality)` - Analyze natural language prompts and create execution plans
- `review_generated_scene(scene_data)` - Review and provide feedback on generated scenes  
- `validate_scene_input(input_data)` - Validate input data format
- `process_scene_task(task_data)` - Process scene generation tasks
- `get_agent_state()` - Get current agent state
- `reset_agent_state()` - Reset agent state
- `update_agent_metadata(key, value)` - Update agent metadata

## Usage Examples

### In ADK Web UI:

1. **Generate a cyberpunk scene:**
   ```
   analyze_scene_prompt("A futuristic cyberpunk city at night with neon signs and flying cars", "cyberpunk", "high")
   ```

2. **Create a medieval landscape:**
   ```
   analyze_scene_prompt("A medieval castle on a hill surrounded by forests", "medieval", "high")
   ```

3. **Get agent status:**
   ```
   get_agent_state()
   ```

## Configuration

### Environment Variables
- `GOOGLE_API_KEY` - Your Google API key for Gemini models
- `GOOGLE_CLOUD_PROJECT` - Your GCP project ID
- `BLENDER_EXECUTABLE_PATH` - Path to Blender executable (optional)
- `BLENDER_ASSETS_PATH` - Path to asset libraries (default: ./assets)
- `BLENDER_OUTPUT_PATH` - Path for generated files (default: ./output)

### Agent Configuration
Edit `config/settings.yaml` to customize agent behavior:
- Model selection (gemini-2.0-flash-exp, gemini-1.5-pro)
- Temperature settings
- Timeout values
- Asset library paths

## Project Structure

```
blender-ai-agent-studio/
├── agents/                 # ADK Agent implementations
│   ├── base_agent.py      # Base ADK Agent class
│   ├── scene_director.py  # Main orchestrator agent
│   └── ...               # Other specialist agents
├── assets/               # 3D asset libraries
├── config/              # Configuration files
├── adk_config.py        # ADK agent configuration
├── adk_main.py          # Main ADK entry point
└── requirements.txt     # Python dependencies
```

## Development

### Adding New Agents
1. Create a new agent class inheriting from `BaseAgent`
2. Implement ADK tools using the `@tool` decorator
3. Register the agent in `adk_config.py`

### Testing
```bash
python adk_main.py
```

This will initialize the agents and show available tools.

## Integration with Vertex AI

The agents are configured to work with Vertex AI's Gemini models. Ensure your GCP project has:
- Vertex AI API enabled
- Proper authentication configured
- Gemini models available in your region

## Troubleshooting

1. **API Key Issues:**
   - Verify `GOOGLE_API_KEY` is set correctly
   - Check API key permissions for Gemini models

2. **ADK Web UI not starting:**
   - Ensure ADK is installed: `pip install google-genai-adk`
   - Check port 8000 is available

3. **Agent Tools not showing:**
   - Verify agents are properly configured in `adk_config.py`
   - Check agent initialization in `adk_main.py`