"""Shared request and media helpers for OpenAI-compatible nodes."""

import base64
import json
import mimetypes
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit

from .config_utils import get_config_section


def parse_json_options(options_json):
    if not options_json or not str(options_json).strip():
        return {}
    try:
        value = json.loads(options_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Advanced options must be valid JSON: {exc}")
    if not isinstance(value, dict):
        raise ValueError("Advanced options JSON must be a top-level object")
    return value


def resolve_endpoint(base_url, api_protocol):
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url cannot be empty")
    if api_protocol not in ("openai-completions", "openai-responses"):
        raise ValueError(f"Unsupported API protocol: {api_protocol}")
    raw = base_url.strip()
    parts = urlsplit(raw)
    path = parts.path.rstrip("/")
    lower_path = path.lower()
    if lower_path.endswith("/chat/completions"):
        if api_protocol != "openai-completions":
            raise ValueError("base_url points to /chat/completions but protocol is openai-responses")
        endpoint_path = path
    elif lower_path.endswith("/responses"):
        if api_protocol != "openai-responses":
            raise ValueError("base_url points to /responses but protocol is openai-completions")
        endpoint_path = path
    else:
        suffix = "/chat/completions" if api_protocol == "openai-completions" else "/responses"
        endpoint_path = f"{path}{suffix}" if path else suffix
    return urlunsplit((parts.scheme, parts.netloc, endpoint_path, parts.query, parts.fragment))


def get_proxy_config(proxy_options=None):
    config = proxy_options if proxy_options is not None else get_config_section("proxy")
    if not isinstance(config, dict) or not config.get("enable", False):
        return None
    proxies = {}
    for key in ("http", "https"):
        value = config.get(key, "")
        if isinstance(value, str) and value.strip():
            proxies[key] = value.strip()
    return proxies or None


def video_to_data_uri(video, max_bytes=64 * 1024 * 1024):
    """Read a ComfyUI VideoInput into a bounded base64 data URI."""
    if video is None or not hasattr(video, "get_stream_source"):
        raise ValueError("Invalid video input")
    source = video.get_stream_source()
    if isinstance(source, BytesIO):
        source.seek(0)
        data = source.read(max_bytes + 1)
    else:
        with open(source, "rb") as handle:
            data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Video exceeds the maximum size of {max_bytes} bytes")
    if isinstance(source, BytesIO):
        source.seek(0)
    mime = getattr(video, "container", None)
    mime = str(mime).lower() if mime else ""
    mime = {"mp4": "video/mp4", "webm": "video/webm", "mkv": "video/x-matroska"}.get(mime, mime)
    if not mime.startswith("video/"):
        mime = "video/mp4"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
