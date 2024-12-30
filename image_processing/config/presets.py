# image_processing/config/effect_presets.py

# from typing import Dict, Any

EFFECT_ORDER = [
    "rgb",
    "color",
    "glitch",
    "chroma",
    "scan",
    "noise",
    "energy",
    "pulse",
    "consciousness",
]

EFFECT_PRESETS = {
    "glitch": {"intensity": 0.5},
    "chroma": {"intensity": 0.5},
    "scan": {"gap": 2, "opacity": 0.5},
    "noise": {"intensity": 0.5},
    "energy": {"intensity": 0.5},
    "pulse": {"intensity": 0.5},
    "consciousness": {"intensity": 0.5},
    "g": {"intensity": 0.5},
    "b": {"intensity": 0.5},
    "impact": {"intensity": "$TETSUO", "font": 70},
    "frames": {"count": 24},
    "fps": {"count": 6},
    "rgb": {"r": 75, "g": 75, "b": 75},
    "rgbalpha": {"insensity": 255},
}

ANIMATION_PRESETS = {
    "cyberpunk": {
        "params": {
            "glitch": {"intensity": (0.3, 0.8)},
            "chroma": {"offset": (0.2, 0.4)},
            "scan": {"gap": (2, 4), "opacity": (0.4, 0.6)},
            "consciousness": {"intensity": (0.3, 0.7)},
        },
        "frames": 30,
        "fps": 24,
        "description": "Cyberpunk-style glitch effect with scanning",
    },
    "psychic": {
        "params": {
            "energy": {"intensity": (0.4, 0.8)},
            "consciousness": {"intensity": (0.5, 0.9)},
            "pulse": {"intensity": (0.2, 0.6)},
        },
        "frames": 30,
        "fps": 24,
        "description": "Psychic energy visualization",
    },
}
