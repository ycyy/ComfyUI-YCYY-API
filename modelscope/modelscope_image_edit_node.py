import json
import os
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from comfy_api.latest import io

from ..utils.config_utils import get_config_section
from ..utils.image_utils import pil_to_tensor, tensor_to_base64_string


class ModelScopeImageEdit(io.ComfyNode):
    """Edit one or more input images with a ModelScope image model."""

    _CONFIG_SECTION = "modelscope-image-edit"
    _DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2511"

    @classmethod
    def _load_models_from_config(cls) -> List[str]:
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
            with open(config_path, "r", encoding="utf-8") as config_file:
                models = json.load(config_file).get(cls._CONFIG_SECTION, {}).get("models")
            if isinstance(models, list) and models:
                return models
        except (OSError, ValueError, TypeError):
            pass
        return [cls._DEFAULT_MODEL]

    @classmethod
    def _load_config_credentials(cls, config_options=None) -> Tuple[str, str, int]:
        """Load edit endpoint credentials, allowing Config Options overrides."""
        config = get_config_section(cls._CONFIG_SECTION)
        if not config:
            raise ValueError(f"Missing '{cls._CONFIG_SECTION}' section in config file")

        config_options = config_options or {}
        base_url = str(config_options.get("base_url") or config.get("base_url") or "").strip()
        api_key = str(config_options.get("api_key") or config.get("api_key") or "").strip()
        timeout = config_options.get("timeout") or config.get("timeout", 300)
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 300
        if not base_url:
            raise ValueError(f"Missing 'base_url' in {cls._CONFIG_SECTION} section")
        if not api_key:
            raise ValueError(f"Missing 'api_key' in {cls._CONFIG_SECTION} section")
        return base_url.rstrip("/"), api_key, timeout

    @classmethod
    def _get_proxy_config(cls, proxy_options=None) -> Optional[Dict[str, str]]:
        if proxy_options is not None:
            if not proxy_options.get("enable", False):
                return None
            proxies = {
                key: proxy_options[key].strip()
                for key in ("http", "https")
                if isinstance(proxy_options.get(key), str) and proxy_options[key].strip()
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
        model_options = cls._load_models_from_config()
        return io.Schema(
            node_id="YCYY_ModelScope_Image_Edit_API",
            display_name="ModelScope Image Edit API",
            category="YCYY/API/image",
            inputs=[
                io.Image.Input(
                    id="image",
                    tooltip="Input image to edit",
                ),
                io.AnyType.Input(
                    id="config_options",
                    optional=True,
                    tooltip="Optional configuration override from YCYY API Config Options",
                ),
                io.AnyType.Input(
                    id="proxy_options",
                    optional=True,
                    tooltip="Optional proxy configuration override from YCYY API Proxy Options",
                ),
                io.String.Input(id="prompt", multiline=True, tooltip="Image editing instruction"),
                io.String.Input(id="negative_prompt", multiline=True, tooltip="Negative prompt"),
                io.Combo.Input(
                    id="model",
                    options=model_options,
                    default=model_options[0],
                    tooltip="Select ModelScope image editing model",
                ),
                io.Int.Input(id="width", min=64, max=2048, default=1024, step=8),
                io.Int.Input(id="height", min=64, max=2048, default=1024, step=8),
                io.Int.Input(id="steps", min=1, max=100, default=30, step=1),
                io.Float.Input(id="guidance", min=1.5, max=20, default=3.5, step=0.1),
                io.Int.Input(
                    id="seed",
                    min=0,
                    max=2147483647,
                    default=0,
                    control_after_generate=True,
                ),
            ],
            outputs=[io.Image.Output(), io.String.Output()],
            description="This node uses the ModelScope API to edit an input image.",
        )

    @classmethod
    def execute(
        cls,
        image,
        prompt,
        negative_prompt,
        model,
        width,
        height,
        steps,
        guidance,
        seed,
        config_options=None,
        proxy_options=None,
    ) -> io.NodeOutput:
        if image is None:
            raise ValueError("image cannot be empty")
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")
        base_url, api_key, timeout = cls._load_config_credentials(config_options)
        return cls._edit_images(
            base_url,
            api_key,
            image,
            prompt,
            negative_prompt,
            model,
            width,
            height,
            steps,
            guidance,
            seed,
            timeout,
            cls._get_proxy_config(proxy_options),
        )

    @classmethod
    def _edit_images(
        cls,
        base_url,
        api_key,
        image,
        prompt,
        negative_prompt,
        model,
        width,
        height,
        steps,
        guidance,
        seed,
        timeout,
        proxies,
    ) -> io.NodeOutput:
        api_url = f"{base_url}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-ModelScope-Async-Mode": "true",
        }
        # ModelScope accepts the source image as an OpenAI-compatible data URL.
        image_data = tensor_to_base64_string(image[0].unsqueeze(0) if image.ndim == 4 else image)
        payload = {
            "model": model,
            "prompt": prompt,
            "image_url": f"data:image/png;base64,{image_data}",
            "size": f"{width}x{height}",
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

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
                base_url, api_key, task_id, timeout, proxies
            )
            output_response = requests.get(output_image_url, timeout=timeout, proxies=proxies)
            output_response.raise_for_status()
            result_image = Image.open(BytesIO(output_response.content)).convert("RGB")
            return io.NodeOutput(
                pil_to_tensor(result_image),
                json.dumps(task_data, ensure_ascii=False),
            )
        except Exception as error:
            raise RuntimeError(
                json.dumps(
                    {"success": False, "message": f"ModelScope image edit failed: {error}"},
                    ensure_ascii=False,
                )
            ) from error

    @classmethod
    def _wait_for_task(cls, base_url, api_key, task_id, timeout, proxies):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = requests.get(
                f"{base_url}/v1/tasks/{task_id}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-ModelScope-Task-Type": "image_generation",
                },
                timeout=timeout,
                proxies=proxies,
            )
            if response.status_code != 200:
                raise RuntimeError(f"Task query HTTP {response.status_code}: {response.text}")
            data = response.json()
            status = data.get("task_status")
            if status == "SUCCEED":
                output_images = data.get("output_images") or []
                if output_images:
                    return output_images[0], data
                raise RuntimeError("Task succeeded without output image")
            if status == "FAILED":
                raise RuntimeError(data.get("message") or "Image editing task failed")
            time.sleep(min(5, max(0, deadline - time.monotonic())))
        raise TimeoutError("Timed out waiting for ModelScope image editing task")

    @classmethod
    def _create_empty_image(cls):
        return torch.zeros(1, 512, 512, 3, dtype=torch.float32)
