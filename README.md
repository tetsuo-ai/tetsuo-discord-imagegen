# Tetimi - Tetsuo Image Effect Bot
A Discord bot that creates cyberpunk-inspired visual effects inspired by Akira and Tetsuo, using intelligent processing to adapt effects based on image content.

## Features
- Content-aware processing system that analyzes and adapts effects
- Dual-layer RGB processing with separate color overlay and colorization effects
- Advanced visual effects:
  - Energy patterns with edge detection and dynamic line placement
  - Pulse effects with adaptive brightness and spatial distribution
  - Enhanced chromatic aberration with natural distortion falloff
  - Intelligent glitch effects with controlled chaos
  - Dynamic scanline generation with glow effects
  - Noise generation with color preservation
  - High-quality ASCII art conversion with upscaling option
  - Silkscreen effect with multi-color separations
- Comprehensive preset system with fine-tuned combinations
- Random image selection from designated folder

## Commands
### Basic Usage
```bash
!image [options] - Process image with specified effects
!image --random - Process random image from folder
!ascii [options] - Generate ASCII art
!ascii up random - Generate upscaled ASCII from random image
!image_help - Display command help
!ascii_help - Display ASCII conversion help
```

### Effect Parameters
- `--preset <name>` - Use preset effect combination
- `--rgb <r> <g> <b> --rgbalpha <0-255>` - Color overlay
- `--color <r> <g> <b> --coloralpha <0-255>` - Deep colorization
- `--glitch <1-50>` - Glitch intensity
- `--chroma <1-40>` - Chromatic aberration strength
- `--scan <1-200>` - Scanline gap
- `--noise <0-2>` - Noise intensity
- `--energy <0-2>` - Energy effect strength
- `--pulse <0-2>` - Pulse effect intensity
- `--silkscreen` - Apply silkscreen effect

### Available Presets
- `cyberpunk`: Cyan-dominant with dual-layer effects
- `vaporwave`: Pink-heavy with strong aberration
- `glitch_art`: Heavy distortion with controlled chaos
- `retro`: Classic scanline look with subtle coloring
- `matrix`: Green-tinted cyberpunk style
- `synthwave`: 80s-inspired magenta effects
- `akira`: Classic Akira-inspired red tones
- `tetsuo`: Intense purple with layered effects
- `neo_tokyo`: Urban cyberpunk with dual RGB processing
- `psychic`: Psychedelic effect combination
- `tetsuo_rage`: Maximum intensity on all effects

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with: `DISCORD_TOKEN=your_token_here`
4. Create `images` folder for input images
5. Run: `python tetimi.py`

## Effect System
The bot employs intelligent processing that analyzes images for optimal effect application:
- Brightness analysis for opacity and pulse placement
- Edge detection for energy pattern placement
- Color variance for chromatic aberration strength
- Image complexity for glitch intensity
- Contrast analysis for noise distribution
- Color preservation in all effects

## Contributing
Issues and pull requests welcome to help improve the bot.

## License
MIT License

## Credits
Created by Graceus777, richinseattle for tetsuo-ai
Inspired by Katsuhiro Otomo's AKIRA
