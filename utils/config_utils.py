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
# 根据配置段 key 获取模型列表
def get_models_list(section_key):
    try:
        section_config = get_config_section(section_key)
        # 验证配置是否存在
        if not section_config:
            raise ValueError(f"Missing {section_key} section in config file")

        # 直接获取models列表
        if 'models' not in section_config:
            raise ValueError("Missing 'models' in section")

        models = section_config['models']

        # 验证models是否为列表且不为空
        if not isinstance(models, list):
            raise ValueError("'models' must be a list")

        if not models:
            raise ValueError("'models' list cannot be empty")
        return models
    except Exception as e:
        raise ValueError(f"Failed to load models: {str(e)}")


OPENAI_PROTOCOLS = {"openai-completions", "openai-responses"}


def get_openai_apis(section_key="openai-text"):
    """Return normalized OpenAI-compatible API configurations.

    ``openai-text`` is normally an array.  A legacy single mapping is also
    accepted so existing installations can migrate without breaking nodes.
    """
    raw = get_config_section(section_key)
    if raw is None:
        raise ValueError(f"Missing {section_key} section in config file")
    if isinstance(raw, dict):
        raw_items = [raw]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raise ValueError(f"{section_key} must be an object or array")
    if not raw_items:
        raise ValueError(f"{section_key} cannot be empty")

    result = []
    names = set()
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"{section_key}[{index}] must be an object")
        name = item.get("api-name") or item.get("api_name")
        # Legacy single-object configurations have no name.  Give them a
        # stable display name while retaining the original fields.
        if not name:
            if len(raw_items) == 1:
                name = "default"
            else:
                raise ValueError(f"{section_key}[{index}] missing 'api-name'")
        name = str(name).strip()
        if not name:
            raise ValueError(f"{section_key}[{index}] api-name cannot be empty")
        if name in names:
            raise ValueError(f"Duplicate api-name: {name}")
        names.add(name)

        base_url = str(item.get("base_url", "")).strip()
        if not base_url:
            raise ValueError(f"{section_key}[{index}] base_url cannot be empty")
        models = item.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError(f"{section_key}[{index}] 'models' must be a non-empty list")
        models = [str(model).strip() for model in models if str(model).strip()]
        if not models:
            raise ValueError(f"{section_key}[{index}] 'models' cannot be empty")
        protocol = item.get("api_protocol", "openai-completions")
        protocol = str(protocol).strip() or "openai-completions"
        if protocol not in OPENAI_PROTOCOLS:
            raise ValueError(f"Unsupported api_protocol '{protocol}' in {section_key}[{index}]")
        timeout = item.get("timeout", 120)
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 120
        if timeout <= 0:
            raise ValueError(f"{section_key}[{index}] timeout must be positive")

        result.append({
            "api-name": name,
            "base_url": base_url,
            "api_key": str(item.get("api_key", "") or "").strip(),
            "timeout": timeout,
            "api_protocol": protocol,
            "models": models,
        })
    return result


def get_api_names(section_key="openai-text"):
    return [item["api-name"] for item in get_openai_apis(section_key)]


def get_api_config(api_name, section_key="openai-text"):
    for item in get_openai_apis(section_key):
        if item["api-name"] == api_name:
            return item
    raise ValueError(f"Unknown API name: {api_name}")


DEFAULT_GROK_MODELS = [
    "grok-imagine-image-2.0",
    "grok-imagine-image-quality",
    "grok-imagine-image-pro",
    "grok-imagine-image",
]


def get_grok_apis(section_key="grok-image"):
    """Return normalized Grok API configurations.

    ``grok-image`` is normally an array. A legacy single mapping is also
    accepted so existing installations can migrate without breaking nodes.
    """
    raw = get_config_section(section_key)
    if raw is None:
        return [{
            "api-name": "default",
            "base_url": "https://api.x.ai/v1",
            "api_key": "",
            "timeout": 120,
            "models": list(DEFAULT_GROK_MODELS),
        }]
    if isinstance(raw, dict):
        raw_items = [raw]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raise ValueError(f"{section_key} must be an object or array")
    if not raw_items:
        raise ValueError(f"{section_key} cannot be empty")

    result = []
    names = set()
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"{section_key}[{index}] must be an object")
        name = item.get("api-name") or item.get("api_name")
        if not name:
            if len(raw_items) == 1:
                name = "default"
            else:
                raise ValueError(f"{section_key}[{index}] missing 'api-name'")
        name = str(name).strip()
        if not name:
            raise ValueError(f"{section_key}[{index}] api-name cannot be empty")
        if name in names:
            raise ValueError(f"Duplicate api-name: {name}")
        names.add(name)

        base_url = str(item.get("base_url", "") or "https://api.x.ai/v1").strip()
        if not base_url:
            base_url = "https://api.x.ai/v1"

        models = item.get("models")
        if not isinstance(models, list) or not models:
            models = list(DEFAULT_GROK_MODELS)
        else:
            models = [str(model).strip() for model in models if str(model).strip()]
            if not models:
                models = list(DEFAULT_GROK_MODELS)

        timeout = item.get("timeout", 120)
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 120
        if timeout <= 0:
            timeout = 120

        result.append({
            "api-name": name,
            "base_url": base_url,
            "api_key": str(item.get("api_key", "") or "").strip(),
            "timeout": timeout,
            "models": models,
        })
    return result


def get_grok_api_names(section_key="grok-image"):
    return [item["api-name"] for item in get_grok_apis(section_key)]


def get_grok_api_config(api_name=None, section_key="grok-image"):
    apis = get_grok_apis(section_key)
    if not apis:
        raise ValueError(f"No configured APIs found in {section_key}")
    if not api_name:
        return apis[0]
    for item in apis:
        if item["api-name"] == api_name:
            return item
    raise ValueError(f"Unknown API name: {api_name}")


DEFAULT_OPENAI_IMAGE_MODELS = [
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
]


def get_openai_image_apis(section_key="openai-image"):
    """Return normalized OpenAI Image API configurations.

    ``openai-image`` is normally an array. A legacy single mapping is also
    accepted so existing installations can migrate without breaking nodes.
    """
    raw = get_config_section(section_key)
    if raw is None:
        return [{
            "api-name": "default",
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "timeout": 120,
            "models": list(DEFAULT_OPENAI_IMAGE_MODELS),
        }]
    if isinstance(raw, dict):
        raw_items = [raw]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raise ValueError(f"{section_key} must be an object or array")
    if not raw_items:
        raise ValueError(f"{section_key} cannot be empty")

    result = []
    names = set()
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"{section_key}[{index}] must be an object")
        name = item.get("api-name") or item.get("api_name")
        if not name:
            if len(raw_items) == 1:
                name = "default"
            else:
                raise ValueError(f"{section_key}[{index}] missing 'api-name'")
        name = str(name).strip()
        if not name:
            raise ValueError(f"{section_key}[{index}] api-name cannot be empty")
        if name in names:
            raise ValueError(f"Duplicate api-name: {name}")
        names.add(name)

        base_url = str(item.get("base_url", "") or "https://api.openai.com/v1").strip()
        if not base_url:
            base_url = "https://api.openai.com/v1"

        models = item.get("models")
        if not isinstance(models, list) or not models:
            models = list(DEFAULT_OPENAI_IMAGE_MODELS)
        else:
            models = [str(model).strip() for model in models if str(model).strip()]
            if not models:
                models = list(DEFAULT_OPENAI_IMAGE_MODELS)

        timeout = item.get("timeout", 120)
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 120
        if timeout <= 0:
            timeout = 120

        result.append({
            "api-name": name,
            "base_url": base_url,
            "api_key": str(item.get("api_key", "") or "").strip(),
            "timeout": timeout,
            "models": models,
        })
    return result


def get_openai_image_api_names(section_key="openai-image"):
    return [item["api-name"] for item in get_openai_image_apis(section_key)]


def get_openai_image_api_config(api_name=None, section_key="openai-image"):
    apis = get_openai_image_apis(section_key)
    if not apis:
        raise ValueError(f"No configured APIs found in {section_key}")
    if not api_name:
        return apis[0]
    for item in apis:
        if item["api-name"] == api_name:
            return item
    raise ValueError(f"Unknown API name: {api_name}")


