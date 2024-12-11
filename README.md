# Tetsuo Discord Bot

A Discord bot that applies cyberpunk-inspired image effects and transformations to images.

## Usage 💻

### Basic Commands
```
!tetsuo --rgb R G B        - Apply RGB color to non-white areas
!tetsuo --color #HEXCODE   - Add hex color overlay
!tetsuo --glitch INTENSITY - Add glitch effect (1-20)
!tetsuo --chroma OFFSET    - Add chromatic aberration (1-20)
!tetsuo --scan GAP         - Add scan lines (1-10)
!tetsuo --noise LEVEL      - Add noise (0.0-1.0)
```

### Preset Commands
```
!tetsuo --preset cyberpunk
!tetsuo --preset vaporwave
!tetsuo --preset glitch_art
!tetsuo --preset retro
!tetsuo --preset matrix
!tetsuo --preset synthwave
```

### Combined Effects
You can stack multiple effects:
```
!tetsuo --rgb 255 0 255 --glitch 15 --scan 2
!tetsuo --preset cyberpunk --noise 0.2
!tetsuo --chroma 10 --glitch 5 --rgb 0 255 255
```

### Utility Commands
- `!tetsuo_presets` - Display all available presets and their parameters
- 🗑️ React to any bot message to delete it

## Examples 🎨

### Epic Glitchwave
```
!tetsuo --rgb 255 0 255 --glitch 15 --chroma 20 --scan 2 --noise 0.15
```

### Cyber Terminal
```
!tetsuo --rgb 0 255 0 --scan 1 --noise 0.05 --glitch 3
```

### Digital Corruption
```
!tetsuo --preset glitch_art --chroma 15
```

## Technical Details 🔧

- Image size limits: 50x50 to 2000x2000 pixels
- Maximum file size: 8MB
- Supported formats: PNG (output)
- Effects are applied in sequential order
- All parameters are validated before processing

## Acknowledgments 🙏

- Inspired by cyberpunk aesthetics and glitch art
- Named after Tetsuo from Akira
- Built with Python, Pillow, and Discord.py

## Disclaimer ⚠️

- Effects are applied in sequence and may have unexpected interactions
- Some combinations may produce intense visual effects
