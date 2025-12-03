import os
import json
import time
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, Dict, List, Tuple
from comfy_api.latest import ComfyExtension, io
from ..utils.config_utils import get_config_section
from ..utils.image_utils import pil_to_tensor

class ModelScopeImage(io.ComfyNode):
    """
    这个节点使用 ModelScope API 生成图像
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
                return ["Tongyi-MAI/Z-Image-Turbo"]

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'modelscope-image' in config and 'models' in config['modelscope-image']:
                models = config['modelscope-image']['models']
                if isinstance(models, list) and len(models) > 0:
                    return models

            return ["Tongyi-MAI/Z-Image-Turbo"]
        except Exception:
            return ["Tongyi-MAI/Z-Image-Turbo"]

    @classmethod
    def _load_config_credentials(cls, config_options=None) -> Tuple[str, str, int]:
        """
        从 config.json 中加载并验证 API 凭据，如果提供了 config_options 则优先使用
        返回 (base_url, api_key, timeout) 元组
        """
        # 如果提供了配置覆盖，则使用覆盖配置
        if config_options is not None:
            base_url = config_options.get('base_url', '').strip()
            api_key = config_options.get('api_key', '').strip()
            timeout = config_options.get('timeout', 300)

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

            # 检查是否存在 modelscope 配置段
            if 'modelscope-image' not in config:
                raise ValueError("Missing 'modelscope-image' section in config file")

            modelscope_image_config = config['modelscope-image']

            # 获取并验证 base_url
            if 'base_url' not in modelscope_image_config:
                raise ValueError("Missing 'base_url' in modelscope-image section")
            base_url = modelscope_image_config['base_url'].strip() if isinstance(modelscope_image_config['base_url'], str) else str(modelscope_image_config['base_url']).strip()
            if not base_url:
                raise ValueError("base_url cannot be empty")

            # 获取并验证 api_key
            if 'api_key' not in modelscope_image_config:
                raise ValueError("Missing 'api_key' in modelscope section")
            api_key = modelscope_image_config['api_key'].strip() if isinstance(modelscope_image_config['api_key'], str) else str(modelscope_image_config['api_key']).strip()
            if not api_key:
                raise ValueError("api_key cannot be empty")

            # 获取 timeout 参数，默认值为 300 秒
            timeout = modelscope_image_config.get('timeout', 300)
            if isinstance(timeout, str):
                try:
                    timeout = int(timeout)
                except ValueError:
                    timeout = 300

            # 如果有配置覆盖，则使用覆盖的值（如果提供了）
            if config_options is not None:
                if config_options.get('base_url', '').strip():
                    base_url = config_options['base_url'].strip()
                if config_options.get('api_key', '').strip():
                    api_key = config_options['api_key'].strip()
                if config_options.get('timeout'):
                    timeout = config_options['timeout']

            return base_url, api_key, timeout

        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Config loading error: {str(e)}")

    @classmethod
    def _get_proxy_config(cls, proxy_options=None) -> Optional[Dict]:
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
        返回一个包含该节点所有信息的模式（schema）
        """
        # 从配置文件加载模型列表
        model_options = cls._load_models_from_config()
        default_model = model_options[0]

        return io.Schema(
            node_id="YCYY_ModelScope_Image_API",
            display_name="ModelScope Image API",
            category="YCYY/API/image",
            inputs=[
                io.String.Input(
                    id="prompt",
                    multiline=True,
                    tooltip="Image generation positive prompt"
                ),
                io.String.Input(
                    id="negative_prompt",
                    multiline=True,
                    tooltip="Image generation negative prompt"
                ),
                io.Combo.Input(
                    id="model",
                    options=model_options,
                    default=default_model,
                    tooltip="Select ModelScope image generation model"
                ),
                io.Int.Input(
                    id="width",
                    min=64,
                    max=2048,
                    default=1024,
                    step=8
                ),
                io.Int.Input(
                    id="height",
                    min=64,
                    max=2048,
                    default=1024,
                    step=8
                ),
                io.Int.Input(
                    id="steps",
                    min=1,
                    max=100,
                    default=30,
                    step=1
                ),
                io.Float.Input(
                    id="guidance",
                    min=1.5,
                    max=20,
                    default=3.5,
                    step=0.1
                ),
                io.AnyType.Input(
                    id="config_options",
                    optional=True,
                    tooltip="Optional configuration override"
                ),
                io.AnyType.Input(
                    id="proxy_options",
                    optional=True,
                    tooltip="Optional proxy configuration override"
                ),
                io.Int.Input(
                    id="seed",
                    min=0,
                    max=2147483647,
                    default=0,
                    control_after_generate=True
                )
            ],
            outputs=[
                io.Image.Output(),
                io.String.Output()
            ],
            description="This node uses the ModelScope API to generate images."
        )

    @classmethod
    def execute(cls, prompt, negative_prompt, model, width, height, steps, guidance, seed, config_options=None, proxy_options=None) -> io.NodeOutput:
        """
        节点执行入口
        """
        base_url, api_key, timeout = cls._load_config_credentials(config_options)
        proxies = cls._get_proxy_config(proxy_options)
        if not prompt or not prompt.strip():
            raise Exception("prompt cannot be empty")
        return cls._generate_images(base_url,api_key,prompt,negative_prompt,model,width,height, steps, guidance, seed,timeout,proxies)
    @classmethod
    def _generate_images(cls,base_url,api_key,prompt,negative_prompt,model,width,height, steps, guidance, seed,timeout,proxies)-> io.NodeOutput:
        # 构建返回参数
        result_image = cls._create_empty_image()
        result_message = json.dumps({
            "success": False,
            "message": "API request returns an error"
        })
        output_image_url = None
        # 构建请求 URL
        api_url = f"{base_url}/v1/images/generations"
        # 构建请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-ModelScope-Async-Mode": "true"
        }
        # 构建请求体
        payload = {
            "model": model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "steps": steps,
            "guidance": guidance,
            "seed": seed
        }
        if negative_prompt is not None and negative_prompt:
            payload["negative_prompt"] = negative_prompt
        try:
            response = requests.post(
                api_url, 
                headers=headers, 
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), 
                timeout=timeout, 
                proxies=proxies
            )
            if response.status_code != 200:
                result_message = json.dumps({
                    "success": False,
                    "message": f"API request returns an error.status_code:{response.status_code}.error_reason:{response.text}"
                })
                raise Exception(result_message)
            task_id = response.json()["task_id"]
            if task_id is not None and task_id:
                while True:
                    result = requests.get(
                        f"{base_url}/v1/tasks/{task_id}",
                        headers={
                            'Authorization': f'Bearer {api_key}',
                            'X-ModelScope-Task-Type': 'image_generation'
                        },
                        timeout=timeout
                    )
                    if result.status_code != 200:
                        result_message = json.dumps({
                            "success": False,
                            "message": f"API request returns an error.status_code:{result.status_code}.error_reason:{result.text}"
                        })
                        raise Exception(result_message)                       
                    data = result.json()
                    if data["task_status"] == "SUCCEED":
                        output_image_url = data["output_images"][0]
                        break
                    elif data["task_status"] == "FAILED":
                        result_message = json.dumps({
                            "success": False,
                            "message": "Image generation failed."
                        })
                        break
                    time.sleep(5)
            output_image_response = requests.get(output_image_url, timeout=timeout)
            pil_image = Image.open(BytesIO(output_image_response.content))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            result_image = pil_to_tensor(pil_image)
            result_message = json.dumps({
                "success": True,
                "message": "Image generation success.",
                "image_url": output_image_url
            })
            return io.NodeOutput(result_image,result_message) 
        except Exception as e:
            raise Exception(result_message)

    # 创建空图像
    @classmethod
    def _create_empty_image(cls):
        try:
            return torch.zeros(1, 512, 512, 3, dtype=torch.float32)
        except Exception as e:
            return None