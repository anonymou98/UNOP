# utils/config.py
"""
Configuration Loader
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config with inheritance support"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    if 'base_config' in config:
        base_path = config_path.parent / config['base_config']
        base_config = load_config(base_path)
        config = merge_configs(base_config, config)
        del config['base_config']
        
    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursive merge"""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def save_config(config: Dict, save_path: str):
    """Save config to YAML"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def validate_config(config: Dict):
    """Validate required fields"""
    required = ['model', 'data', 'training']
    for k in required:
        if k not in config:
            raise ValueError(f"Missing required config section: {k}")
            
    if 'problem_type' not in config['model']:
        raise ValueError("Missing model.problem_type")
        
    return True