import json
import os

# 加载配置文件
def load_config():
    """加载完整的config.json配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), '..',  "config.json")
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise ValueError(f"Config loading error: {str(e)}")

# 根据 key 获取对应的 config 配置段
def get_config_section(section_key):
    """
    根据 key 获取对应的 config 配置段
    Args:
        section_key: 配置段的键名，例如 'ollama-vlm', 'gemini-image'
    Returns:
        对应的配置段字典，如果不存在则返回 None
    """
    try:
        config = load_config()
        return config.get(section_key, None)
    except Exception:
        return None