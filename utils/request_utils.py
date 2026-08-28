"""Shared request and media helpers for OpenAI-compatible nodes."""

import base64
import json
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit

import requests

from .config_utils import get_config_section


class ToolChoiceRejected(RuntimeError):
    """The provider explicitly rejected the request's tool_choice field."""


class FunctionToolsRejected(RuntimeError):
    """The provider explicitly rejected the requested function tool schema."""


def _raise_openai_http_error(response, payload):
    """Raise the same stable error classes for JSON and streaming requests."""
    if 200 <= response.status_code < 300:
        return
    detail = response.text[:1000]
    lower_detail = detail.lower()
    choice_rejected = (
        response.status_code in (400, 404, 415, 422)
        and "tool_choice" in payload
        and any(marker in lower_detail for marker in ("tool_choice", "tool choice"))
        and any(
            marker in lower_detail
            for marker in (
                "unsupported", "unknown", "unrecognized", "invalid",
                "not support", "not allowed", "not permitted", "extra field",
            )
        )
    )
    if choice_rejected:
        raise ToolChoiceRejected(
            f"API request rejected tool_choice ({response.status_code}): {detail}"
        )
    tool_error = response.status_code in (400, 404, 415, 422) and any(
        marker in lower_detail
        for marker in ("tool", "function", "unsupported", "unknown field")
    )
    if payload.get("tools") and tool_error:
        raise FunctionToolsRejected(
            f"API request failed ({response.status_code}); this API/model may not support "
            f"the requested Skill tools: {detail}"
        )
    raise RuntimeError(f"API request failed ({response.status_code}): {detail}")


def post_openai_json(endpoint, headers, payload, timeout, proxies):
    """POST an OpenAI-compatible JSON request with stable error classification."""
    response = requests.post(
        endpoint, headers=headers, json=payload, timeout=timeout, proxies=proxies
    )
    _raise_openai_http_error(response, payload)
    if not response.text.strip():
        raise ValueError("API returned an empty response")
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError("API returned invalid JSON") from exc


def _iter_sse_data(response):
    """Yield complete SSE data payloads, independent of HTTP chunk boundaries."""
    data_lines = []
    # The SSE specification mandates UTF-8. Requests may otherwise infer
    # ISO-8859-1 for text/event-stream responses without a charset.
    response.encoding = "utf-8"
    # requests defaults to 512-byte buffering here. Small token events can
    # otherwise sit in that buffer until generation is nearly complete,
    # making a real SSE response appear non-streaming in the UI.
    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
        if line is None:
            continue
        if isinstance(line, bytes):
            line = line.decode(response.encoding or "utf-8", errors="replace")
        line = line.rstrip("\r")
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def _completion_stream_event(event):
    """Return the provider-neutral meaning of a Chat Completions event."""
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, None, False
    choice = choices[0]
    delta = choice.get("delta")
    delta = delta if isinstance(delta, dict) else {}
    content = delta.get("content")
    reasoning = delta.get("reasoning_content")
    return (
        content if isinstance(content, str) and content else None,
        "reasoning" if isinstance(reasoning, str) and reasoning else None,
        choice.get("finish_reason") is not None,
    )


def post_openai_stream(
    endpoint,
    headers,
    payload,
    timeout,
    proxies,
    protocol,
    on_delta,
    final_response_parser=None,
    on_activity=None,
):
    """POST an OpenAI-compatible SSE request and return its complete text."""
    if protocol not in ("openai-completions", "openai-responses"):
        raise ValueError(f"Unsupported streaming protocol: {protocol}")

    chunks = []
    completed_response = None
    reasoning_notified = False
    completed = False
    stream_headers = {
        **headers,
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        # Some OpenAI-compatible gateways gzip SSE responses and only flush
        # compressed blocks occasionally. Identity encoding keeps token-sized
        # events observable as soon as the provider sends them.
        "Accept-Encoding": "identity",
    }
    with requests.post(
        endpoint,
        headers=stream_headers,
        json=payload,
        timeout=timeout,
        proxies=proxies,
        stream=True,
    ) as response:
        _raise_openai_http_error(response, payload)
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/event-stream" not in content_type:
            raise ValueError("Streaming API returned a non-SSE response")

        for raw_data in _iter_sse_data(response):
            if raw_data.strip() == "[DONE]":
                completed = True
                break
            try:
                event = json.loads(raw_data)
            except json.JSONDecodeError as exc:
                raise ValueError("Streaming API returned invalid SSE JSON") from exc
            if not isinstance(event, dict):
                raise ValueError("Streaming API returned a non-object SSE event")
            if event.get("error"):
                raise ValueError(f"Streaming API failed: {event['error']}")

            if protocol == "openai-completions":
                text, activity, event_completed = _completion_stream_event(event)
                completed = completed or event_completed
            else:
                event_type = event.get("type")
                text = (
                    event.get("delta")
                    if event_type == "response.output_text.delta"
                    else None
                )
                activity = (
                    "reasoning"
                    if event_type == "response.reasoning_text.delta"
                    and isinstance(event.get("delta"), str)
                    and event.get("delta")
                    else None
                )
                if event_type == "response.completed":
                    completed_response = event.get("response")
                    completed = True
                if event_type in ("error", "response.failed", "response.incomplete"):
                    detail = event.get("error") or event.get("response") or event
                    raise ValueError(f"Streaming API failed: {detail}")

            if activity == "reasoning" and not reasoning_notified:
                reasoning_notified = True
                if on_activity is not None:
                    on_activity("reasoning")

            if isinstance(text, str) and text:
                chunks.append(text)
                on_delta(text)

    if not completed:
        raise ValueError("Streaming API ended before a completion marker")

    result = "".join(chunks)
    if (
        protocol == "openai-responses"
        and isinstance(completed_response, dict)
        and final_response_parser is not None
    ):
        try:
            final_result = final_response_parser(completed_response)
        except (TypeError, ValueError):
            if not result:
                raise
        else:
            if isinstance(final_result, str) and final_result.strip():
                result = final_result
    if not isinstance(result, str) or not result.strip():
        if reasoning_notified:
            raise ValueError("Streaming API returned no formal text content")
        raise ValueError("Streaming API returned no text content")
    return result


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
