import json
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from comfy_api.latest import io

try:
    from ..utils.config_utils import (
        DEFAULT_MODELSCOPE_IMAGE_MODELS,
        get_config_section,
        get_modelscope_image_api_config,
        get_modelscope_image_apis,
    )
    from ..utils.image_utils import pil_to_tensor, tensor_to_base64_string
except (ImportError, ValueError):
    from utils.config_utils import (
        DEFAULT_MODELSCOPE_IMAGE_MODELS,
        get_config_section,
        get_modelscope_image_api_config,
        get_modelscope_image_apis,
    )
    from utils.image_utils import pil_to_tensor, tensor_to_base64_string


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/ycyy/modelscope-image/apis/all")
    async def get_all_modelscope_image_apis(request):
        try:
            return web.json_response([
                {"api-name": item["api-name"], "models": item["models"]}
                for item in get_modelscope_image_apis()
            ])
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)
except Exception:
    pass


DEFAULT_MODELS = list(DEFAULT_MODELSCOPE_IMAGE_MODELS)


class ModelScopeImage(io.ComfyNode):
    """Generate or edit an image through the asynchronous ModelScope API."""

    @classmethod
    def _load_models_from_config(cls, api_name: Optional[str] = None) -> List[str]:
        try:
            apis = get_modelscope_image_apis()
            if api_name:
                for item in apis:
                    if item["api-name"] == api_name:
                        return item.get("models") or list(DEFAULT_MODELS)
            models = list(dict.fromkeys(
                model for item in apis for model in item.get("models", [])
            ))
            return models or list(DEFAULT_MODELS)
        except Exception:
            return list(DEFAULT_MODELS)

    @classmethod
    def _load_config_credentials(
        cls,
        api_name: Optional[str] = None,
        config_options: Optional[dict] = None,
    ) -> Tuple[str, str, int]:
        try:
            api_config = get_modelscope_image_api_config(api_name)
        except Exception:
            apis = get_modelscope_image_apis()
            api_config = apis[0] if apis else {
                "base_url": "https://api-inference.modelscope.cn",
                "api_key": "",
                "timeout": 300,
            }

        base_url = str(
            api_config.get("base_url") or "https://api-inference.modelscope.cn"
        ).strip()
        api_key = str(api_config.get("api_key") or "").strip()
        timeout = api_config.get("timeout", 300)

        config_options = config_options or {}
        override_base_url = str(config_options.get("base_url") or "").strip()
        override_api_key = str(config_options.get("api_key") or "").strip()
        if override_base_url:
            base_url = override_base_url
        if override_api_key:
            api_key = override_api_key
        if config_options.get("timeout"):
            timeout = config_options["timeout"]

        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 300
        if timeout <= 0:
            timeout = 300
        if not base_url:
            raise ValueError("ModelScope base_url cannot be empty")
        if not api_key:
            raise ValueError(
                "ModelScope API key not found. Please provide an api_key in "
                "config.json ('modelscope-image') or via API Config Options."
            )
        return base_url, api_key, timeout

    @classmethod
    def _get_proxy_config(
        cls, proxy_options: Optional[dict] = None
    ) -> Optional[Dict[str, str]]:
        if proxy_options is not None:
            if not proxy_options.get("enable", False):
                return None
            proxies = {
                key: proxy_options[key].strip()
                for key in ("http", "https")
                if isinstance(proxy_options.get(key), str)
                and proxy_options[key].strip()
            }
            return proxies or None

        try:
            proxy_config = get_config_section("proxy") or {}
            if not proxy_config.get("enable", False):
                return None
            proxies = {
                key: proxy_config[key]
                for key in ("http", "https")
                if proxy_config.get(key)
            }
            return proxies or None
        except Exception:
            return None

    @classmethod
    def define_schema(cls) -> io.Schema:
        try:
            apis = get_modelscope_image_apis()
            api_names = [item["api-name"] for item in apis]
            models = list(dict.fromkeys(
                model for item in apis for model in item.get("models", [])
            ))
        except Exception:
            api_names = ["default"]
            models = list(DEFAULT_MODELS)
        if not api_names:
            api_names = ["default"]
        if not models:
            models = list(DEFAULT_MODELS)

        return io.Schema(
            node_id="YCYY_ModelScope_Image_API",
            display_name="ModelScope Image API",
            category="YCYY/API/image",
            inputs=[
                io.String.Input(
                    id="prompt",
                    multiline=True,
                    tooltip="Prompt used to generate or edit the image.",
                ),
                io.String.Input(
                    id="negative_prompt",
                    multiline=True,
                    tooltip="Negative prompt.",
                ),
                io.Combo.Input(
                    id="api_name",
                    options=api_names,
                    default=api_names[0],
                    tooltip="Select a ModelScope image API name.",
                ),
                io.Combo.Input(
                    id="model",
                    options=models,
                    default=models[0],
                    tooltip="Select a model from the chosen API name.",
                ),
                io.Int.Input(id="width", min=64, max=2048, default=1024, step=8),
                io.Int.Input(id="height", min=64, max=2048, default=1024, step=8),
                io.Int.Input(id="steps", min=1, max=100, default=30, step=1),
                io.Float.Input(
                    id="guidance", min=1.5, max=20, default=3.5, step=0.1
                ),
                io.Int.Input(
                    id="seed",
                    min=0,
                    max=2147483647,
                    default=0,
                    control_after_generate=True,
                ),
                io.Image.Input(
                    id="image",
                    optional=True,
                    tooltip="Optional source image. Connect it to edit an image; leave it disconnected to generate one.",
                ),
                io.Custom("YCYY_API_CONFIG_OPTIONS").Input(
                    id="config_options",
                    optional=True,
                    tooltip="Optional configuration override from YCYY API Config Options.",
                ),
                io.Custom("YCYY_API_PROXY_OPTIONS").Input(
                    id="proxy_options",
                    optional=True,
                    tooltip="Optional proxy configuration override from YCYY API Proxy Options.",
                ),
            ],
            outputs=[io.Image.Output(), io.String.Output()],
            description="Generate or edit images through the ModelScope API. Connecting an image enables edit mode.",
        )

    @classmethod
    def execute(
        cls,
        prompt,
        negative_prompt,
        model,
        width,
        height,
        steps,
        guidance,
        seed,
        api_name=None,
        image=None,
        config_options=None,
        proxy_options=None,
    ) -> io.NodeOutput:
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        base_url, api_key, timeout = cls._load_config_credentials(
            api_name=api_name, config_options=config_options
        )
        return cls._request_image(
            base_url=base_url,
            api_key=api_key,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            image=image,
            timeout=timeout,
            proxies=cls._get_proxy_config(proxy_options),
        )

    @classmethod
    def _request_image(
        cls,
        base_url,
        api_key,
        prompt,
        negative_prompt,
        model,
        width,
        height,
        steps,
        guidance,
        seed,
        image,
        timeout,
        proxies,
    ) -> io.NodeOutput:
        service_root = cls._normalize_base_url(base_url)
        api_url = f"{service_root}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-ModelScope-Async-Mode": "true",
        }
        mode = "edit" if image is not None else "generation"
        payload = {
            "model": model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if image is not None:
            image_data = tensor_to_base64_string(image)
            payload["image_url"] = f"data:image/png;base64,{image_data}"

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                proxies=proxies,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
            task_id = response.json().get("task_id")
            if not task_id:
                raise RuntimeError("ModelScope response did not contain task_id")

            output_image_url, task_data = cls._wait_for_task(
                service_root, api_key, task_id, timeout, proxies
            )
            output_response = requests.get(
                output_image_url, timeout=timeout, proxies=proxies
            )
            output_response.raise_for_status()
            result_image = Image.open(BytesIO(output_response.content)).convert("RGB")
            result_info = {
                "success": True,
                "message": f"Image {mode} success.",
                "mode": mode,
                "model": model,
                "task_id": task_id,
                "image_url": output_image_url,
                "task": task_data,
            }
            return io.NodeOutput(
                pil_to_tensor(result_image),
                json.dumps(result_info, ensure_ascii=False),
            )
        except Exception as error:
            raise RuntimeError(
                json.dumps(
                    {
                        "success": False,
                        "mode": mode,
                        "message": f"ModelScope image {mode} failed: {error}",
                    },
                    ensure_ascii=False,
                )
            ) from error

    @classmethod
    def _wait_for_task(cls, service_root, api_key, task_id, timeout, proxies):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = requests.get(
                f"{service_root}/v1/tasks/{task_id}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-ModelScope-Task-Type": "image_generation",
                },
                timeout=timeout,
                proxies=proxies,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Task query HTTP {response.status_code}: {response.text}"
                )
            data = response.json()
            status = data.get("task_status")
            if status == "SUCCEED":
                output_images = data.get("output_images") or []
                if output_images:
                    return output_images[0], data
                raise RuntimeError("Task succeeded without output image")
            if status == "FAILED":
                raise RuntimeError(data.get("message") or "Image task failed")
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(5, remaining))
        raise TimeoutError("Timed out waiting for ModelScope image task")

    @classmethod
    def _normalize_base_url(cls, base_url: str) -> str:
        clean_url = base_url.strip().rstrip("/")
        for suffix in ("/v1/images/generations", "/v1/images", "/v1"):
            if clean_url.endswith(suffix):
                clean_url = clean_url[:-len(suffix)]
                break
        if not clean_url:
            raise ValueError("ModelScope base_url cannot be empty")
        return clean_url

    @classmethod
    def _create_empty_image(cls):
        return torch.zeros(1, 512, 512, 3, dtype=torch.float32)
