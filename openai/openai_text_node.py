import json
import hashlib
import requests
from aiohttp import web
from server import PromptServer
from comfy_api.latest import io

from ..utils.config_utils import get_api_config, get_api_names, get_openai_apis
from ..utils.image_utils import tensor_to_base64_string
from ..utils.request_utils import get_proxy_config, resolve_endpoint, video_to_data_uri


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
                io.Combo.Input(id="api_name", options=names, default=names[0]),
                io.Combo.Input(id="model", options=models, default=models[0]),
                io.String.Input(id="system_prompt", multiline=True, default=""),
                io.String.Input(id="user_prompt", multiline=True),
                io.Boolean.Input(id="persist_context", default=True),
                io.Boolean.Input(id="clear_history", default=False),
                io.Image.Input("images", optional=True, tooltip="Optional image input"),
                io.Video.Input("videos", optional=True, tooltip="Optional video input"),
                io.AnyType.Input(id="config_options", optional=True),
                io.AnyType.Input(id="proxy_options", optional=True),
                io.AnyType.Input(id="advanced_options", optional=True),
            ],
            outputs=[
                io.String.Output(id="Result", display_name="Result"),
                io.String.Output(id="Conversation", display_name="Conversation"),
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
    def _parse_completions(cls, response):
        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError('Completions response is missing a non-empty "choices" array')
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise ValueError('Completions response is missing "message"')
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Completions response contains no final text content")
        return content

    @classmethod
    def _parse_responses(cls, response):
        data = response.json()
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
                clear_history=False, images=None, videos=None, config_options=None,
                proxy_options=None, advanced_options=None, unique_id=None) -> io.NodeOutput:
        if not user_prompt or not user_prompt.strip():
            raise ValueError("User prompt cannot be empty")
        api = get_api_config(api_name)
        override = config_options if isinstance(config_options, dict) else {}
        base_url = str(override.get("base_url") or api["base_url"]).strip()
        api_key = str(override.get("api_key") if "api_key" in override else api["api_key"]).strip()
        timeout = override.get("timeout", api["timeout"])
        protocol = override.get("api_protocol", api["api_protocol"])
        endpoint = resolve_endpoint(base_url, protocol)
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = api["timeout"]
        key = cls._session_key(unique_id, endpoint, protocol, model, system_prompt)
        if clear_history:
            cls._conversation_history.pop(key, None)

        if videos is not None:
            video_uri = video_to_data_uri(videos)
        else:
            video_uri = None

        if protocol == "openai-completions":
            history = list(cls._conversation_history.get(key, [])) if persist_context else []
            if not history and system_prompt:
                history.append({"role": "system", "content": system_prompt})
            content = [{"type": "text", "text": user_prompt}] + _image_parts(images, protocol)
            if video_uri:
                content.append({"type": "video_url", "video_url": {"url": video_uri}})
            user_message = {"role": "user", "content": content}
            history.append(user_message)
            payload = {"model": model, "messages": history, "stream": False}
        else:
            history = list(cls._conversation_history.get(key, [])) if persist_context else []
            content = [{"type": "input_text", "text": user_prompt}] + _image_parts(images, protocol)
            if video_uri:
                content.append({"type": "input_video", "video_url": video_uri})
            current = {"role": "user", "content": content}
            history.append(current)
            payload = {"model": model, "input": history, "stream": False}
            if system_prompt:
                payload["instructions"] = system_prompt
        payload = cls._apply_advanced_options(payload, advanced_options, protocol)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = requests.post(endpoint, headers=headers, json=payload,
                                      timeout=timeout, proxies=get_proxy_config(proxy_options))
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(f"API request failed ({response.status_code}): {response.text[:1000]}")
            if not response.text.strip():
                raise ValueError("API returned an empty response")
            result = cls._parse_completions(response) if protocol == "openai-completions" else cls._parse_responses(response)
        except ValueError:
            raise
        except Exception as exc:
            detail = str(exc)
            if videos is not None:
                raise ValueError(f"Video input is not supported by this API or protocol: {detail}")
            raise ValueError(f"The API request failed: {detail}")

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
        return io.NodeOutput(result, conversation)
