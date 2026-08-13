from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[0] # set .../Climademic_Suitability_mdel/ a project root
CONFIG_PATH = PROJECT_ROOT / "config.json"
def load_config():
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    config["paths"] = resolve_paths(config["paths"])
    return config


def resolve_paths(obj):
    if isinstance(obj, dict):
        return {k: resolve_paths(v) for k, v in obj.items()}
    if isinstance(obj, str):
        return PROJECT_ROOT / obj
    return obj