import os
import json
import base64
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple
from comfy_api.latest import io
try:
    from ..utils.config_utils import get_config_section
    from ..utils.image_utils import downscale_image_tensor, common_upscale
except (ImportError, ValueError):
    try:
        from utils.config_utils import get_config_section
        from utils.image_utils import downscale_image_tensor, common_upscale
    except (ImportError, ValueError):
        import sys
        from pathlib import Path
        _root = str(Path(__file__).resolve().parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from utils.config_utils import get_config_section
        from utils.image_utils import downscale_image_tensor, common_upscale


DEFAULT_MODELS = [
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
]


class OpenAIImageAPI(io.ComfyNode):
    """
    OpenAI Image API node for generating and editing images using the gpt-image model series.
    Supports text-to-image generation (/images/generations) and image editing/inpainting (/images/edits).
    """

    @classmethod
    def _load_models_from_config(cls) -> List[str]:
        """
        Load model list from config.json 'openai-image' section.
        Falls back to DEFAULT_MODELS if not configured.
        """
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', "config.json")
            if not os.path.exists(config_path):
                return DEFAULT_MODELS

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'openai-image' in config and 'models' in config['openai-image']:
                models = config['openai-image']['models']
                if isinstance(models, list) and len(models) > 0:
                    clean_models = [str(m).strip() for m in models if str(m).strip()]
                    if clean_models:
                        return clean_models

            return DEFAULT_MODELS
        except Exception:
            return DEFAULT_MODELS

    @classmethod
    def _load_config_credentials(cls, config_options: Optional[dict] = None) -> Tuple[str, str, int]:
        """
        Load API credentials from config_options or config.json.
        Returns (base_url, api_key, timeout) tuple.
        """
        # 1. Check runtime overrides from config_options
        if config_options is not None:
            base_url = str(config_options.get('base_url', '')).strip()
            api_key = str(config_options.get('api_key', '')).strip()
            timeout = config_options.get('timeout', 120)

            if base_url and api_key:
                try:
                    timeout = int(timeout) if int(timeout) > 0 else 120
                except (ValueError, TypeError):
                    timeout = 120
                return base_url, api_key, timeout

        # 2. Check config.json 'openai-image' section
        config_path = os.path.join(os.path.dirname(__file__), '..', "config.json")
        base_url = "https://api.openai.com/v1"
        api_key = ""
        timeout = 120

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                if 'openai-image' in config and isinstance(config['openai-image'], dict):
                    img_cfg = config['openai-image']
                    raw_base = img_cfg.get('base_url', '').strip()
                    if raw_base:
                        base_url = raw_base
                    raw_key = img_cfg.get('api_key', '').strip()
                    if raw_key:
                        api_key = raw_key
                    raw_timeout = img_cfg.get('timeout', 120)
                    try:
                        timeout = int(raw_timeout) if int(raw_timeout) > 0 else 120
                    except (ValueError, TypeError):
                        timeout = 120

                # Fallback to openai-text config if api_key not set
                if not api_key and 'openai-text' in config:
                    text_cfg = config['openai-text']
                    candidates = text_cfg if isinstance(text_cfg, list) else [text_cfg]
                    for item in candidates:
                        if isinstance(item, dict) and item.get('api_key', '').strip():
                            api_key = item['api_key'].strip()
                            # If openai-image base_url was default and text item has a base_url, adopt it
                            if base_url == "https://api.openai.com/v1" and item.get('base_url', '').strip():
                                base_url = item['base_url'].strip()
                            break

            except Exception as e:
                raise ValueError(f"Config loading error: {str(e)}")

        # 3. Check environment variable fallback
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()

        # 4. Apply any partial config_options override
        if config_options is not None:
            if config_options.get('base_url', '').strip():
                base_url = config_options['base_url'].strip()
            if config_options.get('api_key', '').strip():
                api_key = config_options['api_key'].strip()
            if config_options.get('timeout'):
                try:
                    timeout = int(config_options['timeout'])
                except (ValueError, TypeError):
                    pass

        if not api_key:
            raise ValueError("OpenAI API key not found. Please provide an api_key in config.json ('openai-image') or via API Config Options.")

        return base_url, api_key, timeout

    @classmethod
    def _get_proxy_config(cls, proxy_options: Optional[dict] = None) -> Optional[Dict[str, str]]:
        """
        Get proxy settings from proxy_options or config.json.
        """
        if proxy_options is not None:
            if not proxy_options.get('enable', False):
                return None
            proxies = {}
            if proxy_options.get('http', '').strip():
                proxies['http'] = proxy_options['http'].strip()
            if proxy_options.get('https', '').strip():
                proxies['https'] = proxy_options['https'].strip()
            return proxies if proxies else None

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
        model_options = cls._load_models_from_config()
        default_model = model_options[0]

        return io.Schema(
            node_id="YCYY_OpenAI_Image_API",
            display_name="OpenAI Image API",
            category="YCYY/API/image",
            inputs=[
                io.String.Input(
                    id="prompt",
                    multiline=True,
                    tooltip="Text prompt used to generate or edit the image."
                ),
                io.Combo.Input(
                    id="model",
                    options=model_options,
                    default=default_model,
                    tooltip="OpenAI GPT image model."
                ),
                io.Combo.Input(
                    id="size",
                    options=[
                        "auto",
                        "1024x1024",
                        "1024x1536",
                        "1536x1024",
                        "1152x2048",
                        "2048x1152",
                        "2048x2048",
                        "2160x3840",
                        "3840x2160",                       
                        "Custom"
                    ],
                    default="auto",
                    tooltip="Output image size. Select 'Custom' to specify custom width and height."
                ),
                io.Int.Input(
                    id="custom_width",
                    default=1024,
                    min=256,
                    max=3840,
                    step=16,
                    tooltip="Used only when size is 'Custom'. Must be a multiple of 16."
                ),
                io.Int.Input(
                    id="custom_height",
                    default=1024,
                    min=256,
                    max=3840,
                    step=16,
                    tooltip="Used only when size is 'Custom'. Must be a multiple of 16."
                ),
                io.Combo.Input(
                    id="quality",
                    options=[
                        "auto",
                        "low",
                        "medium",
                        "high"
                    ],
                    default="auto",
                    tooltip="Image quality level for GPT image models."
                ),
                io.Combo.Input(
                    id="background",
                    options=[
                        "auto",
                        "opaque",
                        "transparent"
                    ],
                    default="auto",
                    tooltip="Return image with or without background. 'transparent' outputs PNG with alpha transparency."
                ),
                io.Int.Input(
                    id="n",
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
                ),
                io.Image.Input(
                    id="images",
                    optional=True,
                    tooltip="Optional reference image(s) for image editing. GPT image models support up to 16 images."
                ),
                io.Mask.Input(
                    id="mask",
                    optional=True,
                    tooltip="Optional mask for inpainting (white areas will be replaced). Requires exactly one reference image."
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
            ],
            outputs=[
                io.Image.Output(),
                io.String.Output()
            ],
            description="This node uses the OpenAI Image API to generate or edit images using the gpt-image model series."
        )

    @classmethod
    def _resolve_size(cls, size: str, custom_width: int, custom_height: int) -> Optional[str]:
        """
        Validates and resolves the output image size string.
        """
        if size == "Custom":
            if custom_width % 16 != 0 or custom_height % 16 != 0:
                raise ValueError(
                    f"Custom width and height must be multiples of 16, got {custom_width}x{custom_height}"
                )
            if max(custom_width, custom_height) > 3840:
                raise ValueError(
                    f"Custom resolution max edge must be <= 3840, got {custom_width}x{custom_height}"
                )
            min_edge = min(custom_width, custom_height)
            if min_edge <= 0:
                raise ValueError(f"Custom dimensions must be positive, got {custom_width}x{custom_height}")
            ratio = max(custom_width, custom_height) / min_edge
            if ratio > 3.0:
                raise ValueError(
                    f"Custom resolution aspect ratio must not exceed 3:1, got {custom_width}x{custom_height}"
                )
            total_pixels = custom_width * custom_height
            if not (655_360 <= total_pixels <= 8_294_400):
                raise ValueError(
                    f"Custom resolution total pixels must be between 655,360 and 8,294,400, got {total_pixels}"
                )
            return f"{custom_width}x{custom_height}"
        return size

    @classmethod
    def execute(
        cls,
        prompt: str,
        model: str,
        size: str,
        custom_width: int,
        custom_height: int,
        quality: str,
        background: str,
        n: int,
        seed: int,
        images: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        config_options: Optional[dict] = None,
        proxy_options: Optional[dict] = None,
    ) -> io.NodeOutput:
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Load credentials & proxies
        base_url, api_key, timeout = cls._load_config_credentials(config_options)
        proxies = cls._get_proxy_config(proxy_options)

        # Normalize endpoints
        clean_base_url = base_url.rstrip("/")
        for suffix in ("/images/generations", "/images/edits", "/images"):
            if clean_base_url.endswith(suffix):
                clean_base_url = clean_base_url[:-len(suffix)]
                break

        gen_url = f"{clean_base_url}/images/generations"
        edit_url = f"{clean_base_url}/images/edits"

        # Resolve size
        resolved_size = cls._resolve_size(size, custom_width, custom_height)

        if images is not None:
            return cls._edit_images(
                api_url=edit_url,
                api_key=api_key,
                prompt=prompt,
                model=model,
                size=resolved_size,
                quality=quality,
                background=background,
                n=n,
                seed=seed,
                images=images,
                mask=mask,
                timeout=timeout,
                proxies=proxies
            )
        else:
            return cls._generate_images(
                api_url=gen_url,
                api_key=api_key,
                prompt=prompt,
                model=model,
                size=resolved_size,
                quality=quality,
                background=background,
                n=n,
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
        size: str,
        quality: str,
        background: str,
        n: int,
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
            "n": n,
        }
        if size:
            payload["size"] = size
        if quality and quality != "auto":
            payload["quality"] = quality
        if background and background != "auto":
            payload["background"] = background

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            return cls._parse_response(resp, model=model, background=background, timeout=timeout, proxies=proxies)
        except Exception as e:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": f"API request failed. Please check endpoint address and key: {str(e)}"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

    @classmethod
    def _tensor_to_png_bytes(cls, tensor: torch.Tensor) -> bytes:
        """
        Convert torch.Tensor [1, H, W, C] or [H, W, C] to PNG encoded bytes.
        """
        if tensor.ndim == 4:
            tensor = tensor[0]
        tensor_cpu = tensor.cpu()
        channels = tensor_cpu.shape[-1]
        arr = (tensor_cpu.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if channels == 4:
            mode = "RGBA"
        elif channels == 3:
            mode = "RGB"
        elif channels == 1:
            mode = "L"
            arr = arr.squeeze(-1)
        else:
            mode = "RGB"

        pil_img = Image.fromarray(arr, mode=mode)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    @classmethod
    def _edit_images(
        cls,
        api_url: str,
        api_key: str,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        background: str,
        n: int,
        seed: int,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        timeout: int = 120,
        proxies: Optional[dict] = None
    ) -> io.NodeOutput:
        # Split image batch: gpt-image models support up to 16 images
        if len(images.shape) == 4:
            flat_images = [images[i : i + 1] for i in range(images.shape[0])]
        else:
            flat_images = [images.unsqueeze(0)]

        flat_images = flat_images[:16]

        if mask is not None and len(flat_images) != 1:
            raise ValueError("Mask inpainting is only supported when exactly one reference image is provided.")

        # Build multipart files
        files = []
        for i, single_img in enumerate(flat_images):
            # Scale reference image down to <= 2048x2048 if needed
            scaled_img = downscale_image_tensor(single_img, total_pixels=2048 * 2048)
            img_bytes = cls._tensor_to_png_bytes(scaled_img)
            field_name = "image" if len(flat_images) == 1 else "image[]"
            files.append((field_name, (f"image_{i}.png", img_bytes, "image/png")))

        # Process mask if provided
        if mask is not None:
            ref_img = flat_images[0]
            ref_h, ref_w = ref_img.shape[1], ref_img.shape[2]

            cur_mask = mask.squeeze()
            if cur_mask.ndim == 2:
                if cur_mask.shape != (ref_h, ref_w):
                    m_tensor = cur_mask.unsqueeze(0).unsqueeze(0).float()
                    m_tensor = torch.nn.functional.interpolate(
                        m_tensor, size=(ref_h, ref_w), mode="bilinear", align_corners=False
                    )
                    cur_mask = m_tensor.squeeze()

            # OpenAI inpainting specification: transparent alpha areas indicate the region to be modified.
            # ComfyUI masks: white (1.0) is the inpaint region, black (0.0) is the preserved region.
            # Therefore alpha = 1.0 - mask (white area -> alpha 0.0, black area -> alpha 1.0)
            rgba_mask = torch.zeros((ref_h, ref_w, 4), dtype=torch.float32, device="cpu")
            rgba_mask[:, :, 3] = (1.0 - cur_mask.cpu()).clamp(0.0, 1.0)
            scaled_mask = downscale_image_tensor(rgba_mask.unsqueeze(0), total_pixels=2048 * 2048)
            mask_bytes = cls._tensor_to_png_bytes(scaled_mask)
            files.append(("mask", ("mask.png", mask_bytes, "image/png")))

        # Form fields for multipart request
        form_data = {
            "model": model,
            "prompt": prompt,
            "n": str(n),
        }
        if size:
            form_data["size"] = size
        if quality and quality != "auto":
            form_data["quality"] = quality
        if background and background != "auto":
            form_data["background"] = background

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        try:
            resp = requests.post(api_url, headers=headers, files=files, data=form_data, timeout=timeout, proxies=proxies)

            # If the endpoint rejected multipart/form-data with 415 or indicates JSON is expected,
            # fallback to JSON request with base64 data URLs
            if resp.status_code == 415 or (resp.status_code in (400, 422) and "json" in resp.text.lower()):
                return cls._edit_images_json_fallback(
                    api_url=api_url,
                    api_key=api_key,
                    prompt=prompt,
                    model=model,
                    size=size,
                    quality=quality,
                    background=background,
                    n=n,
                    flat_images=flat_images,
                    mask=mask,
                    timeout=timeout,
                    proxies=proxies
                )

            return cls._parse_response(resp, model=model, background=background, timeout=timeout, proxies=proxies)
        except Exception as e:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": f"API request failed. Please check endpoint address and key: {str(e)}"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

    @classmethod
    def _edit_images_json_fallback(
        cls,
        api_url: str,
        api_key: str,
        prompt: str,
        model: str,
        size: str,
        quality: str,
        background: str,
        n: int,
        flat_images: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        timeout: int = 120,
        proxies: Optional[dict] = None
    ) -> io.NodeOutput:
        """
        Fallback for proxies or gateways that only accept JSON with base64 data URLs.
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        input_images = []
        for single_img in flat_images:
            scaled_img = downscale_image_tensor(single_img, total_pixels=2048 * 2048)
            img_b64 = base64.b64encode(cls._tensor_to_png_bytes(scaled_img)).decode("utf-8")
            input_images.append({
                "image_url": f"data:image/png;base64,{img_b64}"
            })

        payload = {
            "model": model,
            "prompt": prompt,
            "images": input_images,
            "n": n,
        }
        if size:
            payload["size"] = size
        if quality and quality != "auto":
            payload["quality"] = quality
        if background and background != "auto":
            payload["background"] = background

        if mask is not None:
            ref_img = flat_images[0]
            ref_h, ref_w = ref_img.shape[1], ref_img.shape[2]
            cur_mask = mask.squeeze()
            if cur_mask.ndim == 2 and cur_mask.shape != (ref_h, ref_w):
                m_tensor = cur_mask.unsqueeze(0).unsqueeze(0).float()
                m_tensor = torch.nn.functional.interpolate(
                    m_tensor, size=(ref_h, ref_w), mode="bilinear", align_corners=False
                )
                cur_mask = m_tensor.squeeze()

            rgba_mask = torch.zeros((ref_h, ref_w, 4), dtype=torch.float32, device="cpu")
            rgba_mask[:, :, 3] = (1.0 - cur_mask.cpu()).clamp(0.0, 1.0)
            scaled_mask = downscale_image_tensor(rgba_mask.unsqueeze(0), total_pixels=2048 * 2048)
            mask_b64 = base64.b64encode(cls._tensor_to_png_bytes(scaled_mask)).decode("utf-8")
            payload["mask"] = {
                "image_url": f"data:image/png;base64,{mask_b64}"
            }

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            return cls._parse_response(resp, model=model, background=background, timeout=timeout, proxies=proxies)
        except Exception as e:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": f"API JSON request failed: {str(e)}"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

    @classmethod
    def _parse_response(
        cls,
        resp: requests.Response,
        model: str = "",
        background: str = "auto",
        timeout: int = 120,
        proxies: Optional[dict] = None
    ) -> io.NodeOutput:
        # Check HTTP status code
        if resp.status_code != 200:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": f"API request error. HTTP {resp.status_code}: {resp.text}"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

        if not resp.text.strip():
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": "API returned an empty response"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

        try:
            data = resp.json()
        except Exception as json_exc:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": f"Failed to parse API JSON response: {str(json_exc)}"
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

        items = data.get("data", [])
        if not items:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": "No image data found in API response",
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
            pil_image = None

            if b64_json:
                try:
                    img_bytes = base64.b64decode(b64_json)
                    pil_image = Image.open(BytesIO(img_bytes))
                except Exception:
                    pass
            elif image_url:
                try:
                    img_resp = requests.get(image_url, timeout=timeout, proxies=proxies)
                    if img_resp.status_code == 200:
                        pil_image = Image.open(BytesIO(img_resp.content))
                except Exception:
                    pass

            if pil_image is not None:
                # If transparent background is requested or image has alpha channel
                if background == "transparent" or (pil_image.mode in ("RGBA", "LA") or "transparency" in pil_image.info):
                    converted = pil_image.convert("RGBA")
                else:
                    converted = pil_image.convert("RGB")

                img_np = np.asarray(converted).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                image_tensors.append(img_tensor)

        if not image_tensors:
            empty_image = cls._create_empty_image()
            err_info = {
                "success": False,
                "message": "Failed to decode or download any image from response",
                "raw_response": data
            }
            return io.NodeOutput(empty_image, json.dumps(err_info, ensure_ascii=False))

        # Ensure consistent channel count across all batch images
        target_channels = image_tensors[0].shape[-1]
        for idx in range(1, len(image_tensors)):
            cur_t = image_tensors[idx]
            if cur_t.shape[-1] != target_channels:
                if target_channels == 4 and cur_t.shape[-1] == 3:
                    # Add opaque alpha channel
                    alpha = torch.ones((*cur_t.shape[:-1], 1), dtype=cur_t.dtype)
                    image_tensors[idx] = torch.cat([cur_t, alpha], dim=-1)
                elif target_channels == 3 and cur_t.shape[-1] == 4:
                    image_tensors[idx] = cur_t[..., :3]

        # Ensure consistent resolution across all batch images (auto size might have slight pixel differences)
        ref_h, ref_w = image_tensors[0].shape[1], image_tensors[0].shape[2]
        for idx in range(1, len(image_tensors)):
            cur_t = image_tensors[idx]
            if cur_t.shape[1] != ref_h or cur_t.shape[2] != ref_w:
                samples = cur_t.movedim(-1, 1)  # [1, C, H, W]
                samples = common_upscale(samples, ref_w, ref_h, "bilinear", "center")
                image_tensors[idx] = samples.movedim(1, -1)

        final_tensor = image_tensors[0] if len(image_tensors) == 1 else torch.cat(image_tensors, dim=0)

        usage = data.get("usage", {})
        info = {
            "success": True,
            "model": model,
            "created": data.get("created"),
            "background": data.get("background", background),
            "size": data.get("size"),
            "quality": data.get("quality"),
            "usage": usage,
            "revised_prompts": revised_prompts
        }
        return io.NodeOutput(final_tensor, json.dumps(info, ensure_ascii=False, indent=2))

    @classmethod
    def _create_empty_image(cls) -> torch.Tensor:
        """Create empty placeholder image on error [1, 512, 512, 3]."""
        return torch.zeros(1, 512, 512, 3, dtype=torch.float32)
