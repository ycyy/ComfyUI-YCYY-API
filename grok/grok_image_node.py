import os
import json
import base64
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple
from comfy_api.latest import ComfyExtension, io
from ..utils.image_utils import tensor_to_base64_string
from ..utils.config_utils import get_config_section


DEFAULT_MODELS = [
    "grok-imagine-image-2.0",
    "grok-imagine-image-quality",
    "grok-imagine-image-pro",
    "grok-imagine-image"
]


class GrokImage(io.ComfyNode):
    """
    这个节点使用 xAI Grok Image API 生成或者修改图片
    """

    @classmethod
    def _load_models_from_config(cls) -> List[str]:
        """
        从 config.json 中加载模型列表
        如果获取不到，返回默认模型列表
        """
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', "config.json")
            if not os.path.exists(config_path):
                return DEFAULT_MODELS

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'grok-image' in config and 'models' in config['grok-image']:
                models = config['grok-image']['models']
                if isinstance(models, list) and len(models) > 0:
                    return models

            return DEFAULT_MODELS
        except Exception:
            return DEFAULT_MODELS

    @classmethod
    def _load_config_credentials(cls, config_options: Optional[dict] = None) -> Tuple[str, str, int]:
        """
        从 config.json 中加载并验证 API 凭据，如果提供了 config_options 则优先使用
        返回 (base_url, api_key, timeout) 元组
        """
        # 如果提供了配置覆盖，则使用覆盖配置
        if config_options is not None:
            base_url = config_options.get('base_url', '').strip()
            api_key = config_options.get('api_key', '').strip()
            timeout = config_options.get('timeout', 120)

            # 如果覆盖配置中有有效的 base_url 和 api_key，则直接返回
            if base_url and api_key:
                return base_url, api_key, timeout

        # 否则从配置文件加载
        config_path = os.path.join(os.path.dirname(__file__), '..', "config.json")

        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 检查是否存在 grok-image 配置段
            if 'grok-image' not in config:
                raise ValueError("Missing 'grok-image' section in config file")

            grok_config = config['grok-image']

            # 获取并验证 base_url (默认 https://api.x.ai/v1)
            base_url = grok_config.get('base_url', 'https://api.x.ai/v1')
            base_url = base_url.strip() if isinstance(base_url, str) else str(base_url).strip()
            if not base_url:
                base_url = "https://api.x.ai/v1"

            # 获取并验证 api_key
            if 'api_key' not in grok_config:
                raise ValueError("Missing 'api_key' in grok-image section")
            api_key = grok_config['api_key'].strip() if isinstance(grok_config['api_key'], str) else str(grok_config['api_key']).strip()
            if not api_key:
                raise ValueError("api_key cannot be empty")

            # 获取 timeout 参数，默认值为 120 秒
            timeout = grok_config.get('timeout', 120)
            if isinstance(timeout, str):
                try:
                    timeout = int(timeout)
                except ValueError:
                    timeout = 120

            # 如果有配置覆盖，则使用覆盖的值（如果提供了）
            if config_options is not None:
                if config_options.get('base_url', '').strip():
                    base_url = config_options['base_url'].strip()
                if config_options.get('api_key', '').strip():
                    api_key = config_options['api_key'].strip()
                if config_options.get('timeout'):
                    timeout = config_options['timeout']

            return base_url, api_key, timeout

        except Exception as e:
            raise ValueError(f"Config loading error: {str(e)}")

    @classmethod
    def _get_proxy_config(cls, proxy_options: Optional[dict] = None) -> Optional[Dict[str, str]]:
        """
        从 config.json 中获取代理配置，如果提供了 proxy_options 则优先使用
        返回 proxies 字典或 None
        """
        # 如果提供了代理覆盖配置
        if proxy_options is not None:
            if not proxy_options.get('enable', False):
                return None

            proxies = {}
            if proxy_options.get('http', '').strip():
                proxies['http'] = proxy_options['http'].strip()
            if proxy_options.get('https', '').strip():
                proxies['https'] = proxy_options['https'].strip()

            return proxies if proxies else None

        # 否则从配置文件加载
        try:
            proxy_config = get_config_section('proxy')
            if not proxy_config or not proxy_config.get('enable', False):
                return None

            proxies = {}
            if proxy_config.get('http'):
                proxies['http'] = proxy_config['http']
            if proxy_config.get('https'):
                proxies['https'] = proxy_config['https']

            return proxies if proxies else None
        except Exception:
            return None

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        返回 GrokImage 节点 schema
        """
        model_options = cls._load_models_from_config()
        default_model = model_options[0]

        return io.Schema(
            node_id="YCYY_Grok_Image_API",
            display_name="Grok Image API",
            category="YCYY/API/image",
            inputs=[
                io.Image.Input(
                    id="images",
                    optional=True,
                    tooltip="Optional image(s) for image-to-image editing. Grok supports up to 3 reference images (1 for pro model)."
                ),
                io.AnyType.Input(
                    id="config_options",
                    optional=True,
                    tooltip="Optional configuration override from YCYY API Config Options"
                ),
                io.AnyType.Input(
                    id="proxy_options",
                    optional=True,
                    tooltip="Optional proxy configuration override from YCYY API Proxy Options"
                ),
                io.String.Input(
                    id="prompt",
                    multiline=True,
                    tooltip="The text prompt used to generate or edit the image"
                ),
                io.Combo.Input(
                    id="model",
                    options=model_options,
                    default=default_model,
                    tooltip="Grok image model"
                ),
                io.Combo.Input(
                    id="aspect_ratio",
                    options=[
                        "auto",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "9:16",
                        "16:9",
                        "9:19.5",
                        "19.5:9",
                        "9:20",
                        "20:9",
                        "1:2",
                        "2:1"
                    ],
                    default="auto",
                    tooltip="Aspect ratio of the output image. 'auto' matches input image in edit mode or generates 1:1."
                ),
                io.Combo.Input(
                    id="resolution",
                    options=[
                        "1K",
                        "2K"
                    ],
                    default="1K",
                    tooltip="Resolution of the output image (1K or 2K)."
                ),
                io.Combo.Input(
                    id="quality",
                    options=[
                        "default",
                        "medium",
                        "low"
                    ],
                    default="medium",
                    tooltip="Quality level, supported only by the grok-imagine-image-2.0 model."
                ),
                io.Int.Input(
                    id="number_of_images",
                    default=1,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Number of images to generate (1 to 10)."
                ),
                io.Int.Input(
                    id="seed",
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    default=0,
                    control_after_generate=True,
                    tooltip="Random seed for generation."
                )
            ],
            outputs=[
                io.Image.Output(),
                io.String.Output()
            ],
            description="This node uses the xAI Grok Image API to generate or edit images."
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        number_of_images: int,
        seed: int,
        images: Optional[torch.Tensor] = None,
        config_options: Optional[dict] = None,
        proxy_options: Optional[dict] = None
    ) -> io.NodeOutput:
        # 加载配置和凭据，如果提供了 config_options 则使用覆盖配置
        base_url, api_key, timeout = cls._load_config_credentials(config_options)
        # 获取代理配置，如果提供了 proxy_options 则使用覆盖配置
        proxies = cls._get_proxy_config(proxy_options)

        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        clean_base_url = base_url.rstrip("/")
        if clean_base_url.endswith("/images/generations") or clean_base_url.endswith("/images/edits"):
            clean_base_url = clean_base_url.rsplit("/images", 1)[0]
        elif clean_base_url.endswith("/images"):
            clean_base_url = clean_base_url.rsplit("/images", 1)[0]

        gen_url = f"{clean_base_url}/images/generations"
        edit_url = f"{clean_base_url}/images/edits"

        if images is not None:
            return cls._edit_images(
                api_url=edit_url,
                api_key=api_key,
                prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                quality=quality,
                number_of_images=number_of_images,
                seed=seed,
                images=images,
                timeout=timeout,
                proxies=proxies
            )
        else:
            return cls._generate_images(
                api_url=gen_url,
                api_key=api_key,
                prompt=prompt,
                model=model,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                quality=quality,
                number_of_images=number_of_images,
                seed=seed,
                timeout=timeout,
                proxies=proxies
            )

    @classmethod
    def _generate_images(
        cls,
        api_url: str,
        api_key: str,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        number_of_images: int,
        seed: int,
        timeout: int,
        proxies: Optional[dict] = None
    ) -> io.NodeOutput:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "n": number_of_images,
            "seed": seed,
            "response_format": "b64_json",
            "resolution": resolution.lower() if resolution else "1k"
        }
        if aspect_ratio != "auto":
            payload["aspect_ratio"] = aspect_ratio
        if quality and quality != "default":
            payload["quality"] = quality

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            return cls._parse_response(resp, model=model, timeout=timeout, proxies=proxies)
        except Exception as e:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": f"The API request failed. Please check if the interface address and key are correct: {str(e)}"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

    @classmethod
    def _edit_images(
        cls,
        api_url: str,
        api_key: str,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        number_of_images: int,
        seed: int,
        images: torch.Tensor,
        timeout: int,
        proxies: Optional[dict] = None
    ) -> io.NodeOutput:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        input_images = []
        total_imgs = images.shape[0] if len(images.shape) >= 4 else 1
        max_imgs = 1 if "pro" in model else 3
        num_to_take = min(total_imgs, max_imgs)

        for idx in range(num_to_take):
            img_tensor = images[idx].unsqueeze(0) if len(images.shape) >= 4 else images.unsqueeze(0)
            b64_str = tensor_to_base64_string(img_tensor, mime_type="image/png")
            input_images.append({
                "url": f"data:image/png;base64,{b64_str}"
            })

        payload = {
            "model": model,
            "prompt": prompt,
            "images": input_images,
            "n": number_of_images,
            "seed": seed,
            "response_format": "b64_json",
            "resolution": resolution.lower() if resolution else "1k"
        }
        if aspect_ratio != "auto":
            payload["aspect_ratio"] = aspect_ratio

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            return cls._parse_response(resp, model=model, timeout=timeout, proxies=proxies)
        except Exception as e:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": f"The API request failed. Please check if the interface address and key are correct: {str(e)}"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

    @classmethod
    def _parse_response(
        cls,
        resp: requests.Response,
        model: str = "",
        timeout: int = 120,
        proxies: Optional[dict] = None
    ) -> io.NodeOutput:
        # 检查 HTTP 状态码
        if resp.status_code != 200:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": f"API request returns an error. status_code: {resp.status_code}, error_reason: {resp.text}"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

        # 检查返回内容是否为空
        if not resp.text.strip():
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": "The API returns an empty content"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

        try:
            data = resp.json()
        except Exception as json_exception:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": f"The API returned a JSON parsing failure: {str(json_exception)}"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

        # 提取图像数据列表
        items = data.get("data", [])
        if not items:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": "Image data not found in response",
                "raw_response": data
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

        image_tensors = []
        revised_prompts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("revised_prompt"):
                revised_prompts.append(item["revised_prompt"])

            b64_json = item.get("b64_json")
            image_url = item.get("url")

            if b64_json:
                try:
                    image_bytes = base64.b64decode(b64_json)
                    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    img_np = np.array(pil_image).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                    image_tensors.append(img_tensor)
                except Exception:
                    pass
            elif image_url:
                try:
                    img_resp = requests.get(image_url, timeout=timeout, proxies=proxies)
                    if img_resp.status_code == 200:
                        pil_image = Image.open(BytesIO(img_resp.content)).convert("RGB")
                        img_np = np.array(pil_image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                        image_tensors.append(img_tensor)
                except Exception:
                    pass

        if not image_tensors:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": "Failed to decode or download any image from response",
                "raw_response": data
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

        if len(image_tensors) == 1:
            final_tensor = image_tensors[0]
        else:
            final_tensor = torch.cat(image_tensors, dim=0)

        usage = data.get("usage", {})
        info = {
            "success": True,
            "model": model,
            "created": data.get("created"),
            "usage": usage,
            "revised_prompts": revised_prompts
        }
        return io.NodeOutput(final_tensor, json.dumps(info, ensure_ascii=False, indent=2))

    @classmethod
    def _create_empty_image(cls) -> torch.Tensor:
        """创建空图像"""
        try:
            return torch.zeros(1, 512, 512, 3, dtype=torch.float32)
        except Exception:
            return None
