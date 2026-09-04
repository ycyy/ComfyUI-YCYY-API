import json
import hashlib
import re
from uuid import uuid4
from aiohttp import web
from server import PromptServer
from comfy_api.latest import io

from ..utils.config_utils import get_api_config, get_openai_apis
from ..utils.image_utils import tensor_to_base64_string
from ..utils.request_utils import (
    get_proxy_config,
    post_openai_json,
    post_openai_sse_events,
    post_openai_stream,
    resolve_endpoint,
    video_to_data_uri,
)
from ..utils.skill_utils import SkillRequestContext


STREAM_EVENT = "ycyy_openai_text_stream"


def _safe_stream_error(exc):
    detail = str(exc).replace("\r", " ").replace("\n", " ")
    detail = re.sub(r"(?i)bearer\s+[a-z0-9._~+/-]+", "Bearer <redacted>", detail)
    detail = re.sub(r"(?i)(api[_-]?key[=:]\s*)[^\s&]+", r"\1<redacted>", detail)
    detail = re.sub(r"\bsk-[a-zA-Z0-9_-]{8,}\b", "sk-<redacted>", detail)
    return detail[:300]


class _TextStreamSink:
    """Route text deltas to the executing ComfyUI client immediately."""

    def __init__(self, node_id):
        try:
            from comfy_execution.utils import get_executing_context
            context = get_executing_context()
        except (ImportError, RuntimeError):
            context = None
        server = PromptServer.instance
        context_node_id = getattr(context, "node_id", None)
        effective_node_id = node_id if node_id is not None else context_node_id
        self.node_id = str(effective_node_id) if effective_node_id is not None else ""
        self.prompt_id = getattr(context, "prompt_id", None)
        self.run_id = uuid4().hex
        self.client_id = getattr(server, "client_id", None)
        self.seq = 0
        self.last_activity = None
        self.current_round = 0

    def _send(self, phase, **extra):
        PromptServer.instance.send_sync(
            STREAM_EVENT,
            {
                "node_id": self.node_id,
                "prompt_id": self.prompt_id,
                "run_id": self.run_id,
                "seq": self.seq,
                "phase": phase,
                **extra,
            },
            self.client_id,
        )
        self.seq += 1

    def start(self):
        self._send("start")

    def delta(self, value):
        if value:
            self._send("delta", delta=value)

    def activity(self, kind, detail=None):
        marker = (kind, detail or "")
        if kind and marker != self.last_activity:
            self.last_activity = marker
            extra = {"activity": kind}
            if detail:
                extra["detail"] = detail
            self._send("activity", **extra)

    def round_start(self, round_index, tools_enabled=True):
        self.current_round = int(round_index)
        self._send("round_start", round=int(round_index), tools_enabled=bool(tools_enabled))

    def candidate_delta(self, value, round_index=None):
        if value:
            extra = {"delta": value}
            extra["round"] = int(self.current_round if round_index is None else round_index)
            self._send("candidate_delta", **extra)

    def tool_call_start(self, call_id, name, path=None, round_index=None):
        extra = {"call_id": call_id, "tool": name}
        if path:
            extra["path"] = path
        extra["round"] = int(self.current_round if round_index is None else round_index)
        self._send("tool_call_start", **extra)

    def tool_call_end(self, call_id, name, status="success", path=None, round_index=None):
        extra = {"call_id": call_id, "tool": name, "status": status}
        if path:
            extra["path"] = path
        extra["round"] = int(self.current_round if round_index is None else round_index)
        self._send("tool_call_end", **extra)

    def round_end(self, round_index, has_tool_calls):
        self._send(
            "round_end",
            round=int(round_index),
            has_tool_calls=bool(has_tool_calls),
            text_status="candidate" if has_tool_calls else "final",
        )

    def end(self, value):
        self._send("end", text=value, stop_reason="stop", text_status="final")

    def error(self, exc):
        self._send("error", message=_safe_stream_error(exc), stop_reason="error", text_status="error")


@PromptServer.instance.routes.get("/ycyy/openai/apis/all")
async def get_all_openai_apis(request):
    try:
        return web.json_response([
            {"api-name": item["api-name"], "models": item["models"]}
            for item in get_openai_apis()
        ])
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


def _image_parts(images, protocol):
    if images is None:
        return []
    parts = []
    for index in range(images.shape[0]):
        data_uri = f"data:image/png;base64,{tensor_to_base64_string(images[index].unsqueeze(0))}"
        if protocol == "openai-completions":
            parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        else:
            parts.append({"type": "input_image", "image_url": data_uri})
    return parts


class OpenAITextAPI(io.ComfyNode):
    _conversation_history = {}
    _max_history_items = 40

    @classmethod
    def _runtime_node_id(cls, explicit_id=None):
        """Resolve the client graph node id for ComfyUI V3 and direct tests."""
        if explicit_id is not None:
            return explicit_id
        hidden_id = getattr(getattr(cls, "hidden", None), "unique_id", None)
        if hidden_id is not None:
            return hidden_id
        try:
            from comfy_execution.utils import get_executing_context
            context = get_executing_context()
        except (ImportError, RuntimeError):
            context = None
        return getattr(context, "node_id", None)

    @classmethod
    def define_schema(cls) -> io.Schema:
        apis = get_openai_apis()
        names = [item["api-name"] for item in apis]
        # The frontend narrows this list when api_name changes.  Keep the
        # server-side schema as the union so ComfyUI validation accepts a
        # model selected from any configured API.
        models = list(dict.fromkeys(model for item in apis for model in item["models"]))
        return io.Schema(
            node_id="YCYY_OpenAI_Text_API",
            display_name="OpenAI Text API",
            category="YCYY/API/text",
            inputs=[
                io.String.Input(id="system_prompt", multiline=True, default=""),
                io.String.Input(id="user_prompt", multiline=True),
                io.Combo.Input(id="api_name", options=names, default=names[0]),
                io.Combo.Input(id="model", options=models, default=models[0]),
                io.Boolean.Input(id="persist_context", default=True),
                io.Boolean.Input(id="clear_history", default=False),
                io.Boolean.Input(
                    id="stream",
                    default=True,
                    tooltip=(
                        "If true, the model response is streamed to the client as it is "
                        "generated using server-sent events (SSE)."
                    ),
                ),
                io.Image.Input("images", optional=True, tooltip="Optional image input"),
                io.Video.Input("videos", optional=True, tooltip="Optional video input"),
                io.AnyType.Input(id="config_options", optional=True),
                io.AnyType.Input(id="proxy_options", optional=True),
                io.AnyType.Input(id="advanced_options", optional=True),
                io.AnyType.Input(
                    id="skill_options",
                    optional=True,
                    tooltip="Optional input from OpenAI Text Skill Options",
                ),
            ],
            outputs=[
                io.String.Output(id="Result", display_name="Result"),
                io.String.Output(id="Conversation", display_name="Conversation"),
                io.String.Output(id="SkillTrace", display_name="Skill Trace"),
            ],
            hidden=[io.Hidden.unique_id],
            description="OpenAI and OpenAI-compatible text, image and video API.",
        )

    @classmethod
    def _apply_advanced_options(cls, payload, options, protocol):
        if not isinstance(options, dict):
            return payload
        protected = {
            "model", "messages", "input", "instructions", "stream",
            "api_key", "base_url", "timeout",
        }
        forbidden = protected.intersection(options)
        if forbidden:
            raise ValueError(f"Advanced options cannot override: {', '.join(sorted(forbidden))}")
        protocol_errors = []
        if protocol == "openai-completions":
            if "max_output_tokens" in options:
                protocol_errors.append("use max_completion_tokens for openai-completions")
            if "reasoning" in options:
                protocol_errors.append("use reasoning_effort for openai-completions")
        else:
            if "max_completion_tokens" in options:
                protocol_errors.append("use max_output_tokens for openai-responses")
            if "reasoning_effort" in options:
                protocol_errors.append("use reasoning for openai-responses")
        if protocol_errors:
            raise ValueError("Invalid advanced options: " + "; ".join(protocol_errors))
        payload.update(options)
        return payload

    @classmethod
    def _session_key(cls, unique_id, api_url, protocol, model, system_prompt):
        raw = json.dumps([unique_id or "", api_url, protocol, model, system_prompt or ""], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _resolve_api_settings(cls, api, config_options):
        """Apply only non-empty, valid external overrides to an API config."""
        override = config_options if isinstance(config_options, dict) else {}

        base_override = override.get("base_url")
        base_url = (
            base_override.strip()
            if isinstance(base_override, str) and base_override.strip()
            else str(api["base_url"]).strip()
        )

        key_override = override.get("api_key")
        api_key = (
            key_override.strip()
            if isinstance(key_override, str) and key_override.strip()
            else str(api["api_key"]).strip()
        )

        protocol_override = override.get("api_protocol")
        protocol = (
            protocol_override.strip()
            if isinstance(protocol_override, str)
            and protocol_override.strip()
            and protocol_override.strip() != "inherit"
            else api["api_protocol"]
        )

        timeout = api["timeout"]
        timeout_override = override.get("timeout")
        if timeout_override not in (None, "") and not isinstance(timeout_override, bool):
            try:
                candidate = int(timeout_override)
                if candidate > 0:
                    timeout = candidate
            except (TypeError, ValueError):
                pass
        return base_url, api_key, timeout, protocol

    @classmethod
    def _completion_message(cls, data):
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError('Completions response is missing a non-empty "choices" array')
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise ValueError('Completions response is missing "message"')
        return message

    @classmethod
    def _parse_completions(cls, data):
        message = cls._completion_message(data)
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Completions response contains no final text content")
        return content

    @classmethod
    def _parse_responses(cls, data):
        status = data.get("status")
        if status != "completed":
            detail = data.get("incomplete_details") or data.get("error") or status or "unknown"
            raise ValueError(f"Responses request did not complete: {detail}")
        chunks = []
        for item in data.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for block in item.get("content", []):
                if isinstance(block, dict) and block.get("type") == "output_text" and block.get("text"):
                    chunks.append(block["text"])
        if not chunks:
            raise ValueError("Responses response contains no final output_text")
        return "\n".join(chunks)

    @classmethod
    def execute(cls, api_name, model, system_prompt, user_prompt, persist_context,
                clear_history=False, stream=False, images=None, videos=None, config_options=None,
                proxy_options=None, advanced_options=None, skill_options=None,
                unique_id=None) -> io.NodeOutput:
        if not user_prompt or not user_prompt.strip():
            raise ValueError("User prompt cannot be empty")
        api = get_api_config(api_name)
        base_url, api_key, timeout, protocol = cls._resolve_api_settings(api, config_options)
        endpoint = resolve_endpoint(base_url, protocol)
        skill = SkillRequestContext.create(skill_options, protocol)
        stream_enabled = bool(stream)
        runtime_node_id = cls._runtime_node_id(unique_id)
        session_prompt = skill.session_discriminator(system_prompt)
        key = cls._session_key(
            runtime_node_id, endpoint, protocol, model, session_prompt,
        )
        if clear_history:
            cls._conversation_history.pop(key, None)
            skill.clear_session(key)

        if videos is not None:
            video_uri = video_to_data_uri(videos)
        else:
            video_uri = None

        if protocol == "openai-completions":
            history = list(cls._conversation_history.get(key, [])) if persist_context and not skill.enabled else []
            request_history = [] if skill.enabled else history
            if not request_history and system_prompt:
                request_history.append({"role": "system", "content": system_prompt})
            content = [{"type": "text", "text": user_prompt}] + _image_parts(images, protocol)
            if video_uri:
                content.append({"type": "video_url", "video_url": {"url": video_uri}})
            user_message = {"role": "user", "content": content}
            if skill.enabled:
                request_history.append(user_message)
            else:
                history.append(user_message)
            payload = {"model": model, "messages": request_history, "stream": stream_enabled}
        else:
            history = list(cls._conversation_history.get(key, [])) if persist_context and not skill.enabled else []
            content = [{"type": "input_text", "text": user_prompt}] + _image_parts(images, protocol)
            if video_uri:
                content.append({"type": "input_video", "video_url": video_uri})
            current = {"role": "user", "content": content}
            history.append(current)
            if system_prompt:
                instructions = system_prompt
            else:
                instructions = None
            payload = {"model": model, "input": history, "stream": stream_enabled}
            if instructions:
                payload["instructions"] = instructions
        skill.validate_advanced_options(advanced_options)
        payload = cls._apply_advanced_options(payload, advanced_options, protocol)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            proxies = get_proxy_config(proxy_options)
            if skill.enabled:
                sink = _TextStreamSink(runtime_node_id) if stream_enabled else None
                if sink is not None:
                    sink.start()
                try:
                    result, conversation = skill.execute(
                        post_openai_json,
                        endpoint,
                        headers,
                        timeout,
                        proxies,
                        payload,
                        key,
                        persist_context=persist_context,
                        stream=stream_enabled,
                        post_stream=post_openai_sse_events,
                        on_delta=sink.candidate_delta if sink is not None else None,
                        on_activity=sink.activity if sink is not None else None,
                        on_round_start=sink.round_start if sink is not None else None,
                        on_round_end=sink.round_end if sink is not None else None,
                        on_tool_call_start=sink.tool_call_start if sink is not None else None,
                        on_tool_call_end=sink.tool_call_end if sink is not None else None,
                    )
                except Exception as exc:
                    if sink is not None:
                        sink.error(exc)
                    raise
                else:
                    if sink is not None:
                        sink.end(result)
            elif stream_enabled:
                sink = _TextStreamSink(runtime_node_id)
                sink.start()
                try:
                    result = post_openai_stream(
                        endpoint,
                        headers,
                        payload,
                        timeout,
                        proxies,
                        protocol,
                        sink.delta,
                        final_response_parser=cls._parse_responses,
                        on_activity=sink.activity,
                    )
                except Exception as exc:
                    sink.error(exc)
                    raise
                else:
                    sink.end(result)
            else:
                data = post_openai_json(endpoint, headers, payload, timeout, proxies)
                result = cls._parse_completions(data) if protocol == "openai-completions" else cls._parse_responses(data)
        except Exception as exc:
            detail = str(exc)
            if isinstance(exc, ValueError):
                raise
            if videos is not None:
                raise ValueError(f"Video input is not supported by this API or protocol: {detail}")
            raise ValueError(f"The API request failed: {detail}")

        if skill.enabled:
            return io.NodeOutput(
                result,
                json.dumps(conversation, ensure_ascii=False),
                skill.trace_json(),
            )

        if persist_context:
            if protocol == "openai-completions":
                stored = history + [{"role": "assistant", "content": result}]
            else:
                # Responses output blocks are not valid input blocks on the
                # next request; replay the assistant turn as input_text.
                stored = history + [{"role": "assistant", "content": [{"type": "input_text", "text": result}]}]
            cls._conversation_history[key] = stored[-cls._max_history_items:]
            conversation = json.dumps(cls._conversation_history[key], ensure_ascii=False)
        else:
            conversation = "[]"
        return io.NodeOutput(
            result,
            conversation,
            skill.trace_json(),
        )
