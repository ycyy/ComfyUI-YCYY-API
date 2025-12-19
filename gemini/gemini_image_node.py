import io
import os
import sys
import json
import base64
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from ..utils.image_utils import tensor_to_base64_string
from ..utils.config_utils import get_config_section

class GeminiImage(io.ComfyNode):
    """
    这个节点使用谷歌Gemini Image API 生成或者修改图片
    """

    @classmethod
    def _load_models_from_config(cls):
        """
        从config.json中加载模型列表
        如果获取不到，返回默认模型列表
        """
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', "config.json")
            if not os.path.exists(config_path):
                return ["gemini-2.5-flash-image"]

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'gemini-image' in config and 'models' in config['gemini-image']:
                models = config['gemini-image']['models']
                if isinstance(models, list) and len(models) > 0:
                    return models

            return ["gemini-2.5-flash-image"]
        except Exception:
            return ["gemini-2.5-flash-image"]

    @classmethod
    def _load_config_credentials(cls, config_options=None):
        """
        从config.json中加载并验证API凭据，如果提供了config_options则优先使用
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
        config_path = os.path.join(os.path.dirname(__file__), '..',  "config.json")

        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 检查是否存在gemini配置段
            if 'gemini-image' not in config:
                raise ValueError("Missing 'gemini-image' section in config file")

            gemini_config = config['gemini-image']

            # 获取并验证base_url
            if 'base_url' not in gemini_config:
                raise ValueError("Missing 'base_url' in gemini-image section")
            base_url = gemini_config['base_url'].strip() if isinstance(gemini_config['base_url'], str) else str(gemini_config['base_url']).strip()
            if not base_url:
                raise ValueError("base_url cannot be empty")

            # 获取并验证api_key
            if 'api_key' not in gemini_config:
                raise ValueError("Missing 'api_key' in gemini-image section")
            api_key = gemini_config['api_key'].strip() if isinstance(gemini_config['api_key'], str) else str(gemini_config['api_key']).strip()
            if not api_key:
                raise ValueError("api_key cannot be empty")

            # 获取timeout参数，默认值为120秒
            timeout = gemini_config.get('timeout', 120)
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
    def _get_proxy_config(cls, proxy_options=None):
        """
        从config.json中获取代理配置，如果提供了proxy_options则优先使用
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
            返回一个包含该节点所有信息的模式（schema）。
            一些可用类型："Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo"。
            对于输出，应使用 "io.Model.Output"，对于输入，可以使用 "io.Model.Input"。
            类型可以是 "Combo" —— 这将是一个供选择的列表。
        """
        # 从配置文件加载模型列表
        model_options = cls._load_models_from_config()
        default_model = model_options[0]

        return io.Schema(
            node_id="YCYY_Gemini_Image_API",
            display_name="Gemini Image API",
            category="YCYY/API/image",
            inputs=[
                io.Image.Input(
                    id="images",
                    optional=True,
                    tooltip="Optional image(s) to use as context for the model"
                ),
                io.AnyType.Input(
                    id="config_options",
                    optional=True,
                    tooltip="Optional configuration override from YCYY API Config Options"
                ),
                io.AnyType.Input(
                    id="proxy_options",
                    optional=True,
                    tooltip="Optional proxy configuration override from YCYY API Proxy Config Options"
                ),
                io.String.Input(
                    id="prompt",
                    multiline=True,
                ),
                io.Combo.Input(
                    id="model",
                    options=model_options,
                    default=default_model
                ),
                io.Combo.Input(
                    id="aspectRatio",
                    options=[
                        "auto",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9"
                    ],
                    default="auto",
                    tooltip="The model defaults to matching the output image size to that of your input image, or otherwise generates 1:1 squares. You can control the aspect ratio of the output image using the aspect ratio"
                ),
                io.Combo.Input(
                    id="imageSize",
                    options=[
                        "1K",
                        "2K",
                        "4K"
                    ],
                    default="1K",
                    tooltip="Control the resolution of the output image. 1K is approximately 1024x1024, 2K is approximately 2048x2048, 4K is approximately 4096x4096(Only effective for the gemini-3 model)."
                ),
                io.Boolean.Input(
                    id="enableSearch",
                    default=False,
                    tooltip="Use the Google Search tool to generate images based on real-time information(Only effective for the gemini-3 model)"
                ),
                io.Int.Input(
                    id="seed",
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    default=0,
                    control_after_generate=True
                )
            ],
            outputs=[
                io.Image.Output(),
                io.String.Output()
            ],
            description="This node uses the Google Gemini Image API to generate or modify images."
        )

    # @classmethod
    # def check_lazy_status(cls, image, string_field, int_field, float_field, print_to_screen):
    #     """
    #         返回一个需要被求值的输入名称列表。

    #         如果存在任何尚未被求值的惰性输入（lazy inputs），此函数将被调用。
    #         只要你返回的列表中至少有一个尚未被求值的字段（并且还有更多未求值的字段存在），
    #         那么一旦请求的字段值可用，此函数将再次被调用。

    #         任何已被求值的输入都将作为参数传递给此函数。任何未被求值的输入的值将为 None。
    #     """
    #     if print_to_screen == "enable":
    #         return ["int_field", "float_field", "string_field"]
    #     else:
    #         return []
    # 执行 GeminiImage 节点
    @classmethod
    def execute(cls, prompt, model, aspectRatio, imageSize, enableSearch, seed, images=None, config_options=None, proxy_options=None) -> io.NodeOutput:
        # 加载配置和凭据，如果提供了config_options则使用覆盖配置
        base_url, api_key, timeout = cls._load_config_credentials(config_options)
        # 获取代理配置，如果提供了proxy_options则使用覆盖配置
        proxies = cls._get_proxy_config(proxy_options)
        if not prompt:
            raise ValueError("prompt cannot be empty")
        api_url = base_url+"/"+model+":generateContent"
        if images is not None:
            return cls._edit_images(api_url,api_key,prompt,model,aspectRatio,imageSize,enableSearch,seed,images,timeout,proxies)
        else:
            return cls._generate_images(api_url,api_key,prompt,model,aspectRatio,imageSize,enableSearch,seed,timeout,proxies)
    # 图生图模式
    @classmethod
    def _edit_images(cls,api_url,api_key,prompt,model,aspectRatio,imageSize,enableSearch,seed,images,timeout,proxies=None)-> io.NodeOutput:
        image_parts = cls._create_image_parts(images)
        image_parts.append(
            {
                "text": prompt
            }
        )
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {
                    "parts": image_parts
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
        # 根据模型类型添加imageConfig - gemini-2.5-flash-image不支持imageSize参数
        if model == "gemini-2.5-flash-image":
            # 对于不支持imageSize的模型，只添加aspectRatio（如果非auto）
            if aspectRatio != "auto":
                payload["generationConfig"]["imageConfig"] = {
                    "aspectRatio": aspectRatio
                }
        elif model == "gemini-3-pro-image-preview":
            # 对于支持imageSize的模型，添加imageSize和aspectRatio
            image_config = {
                "imageSize": imageSize
            }
            if aspectRatio != "auto":
                image_config["aspectRatio"] = aspectRatio
            payload["generationConfig"]["imageConfig"] = image_config
            if enableSearch:
                payload["tools"] = [{"google_search": {}}]

        # print(f"正在请求Gemini文生图API: {api_url}")
        # print(f"请求载荷: {json.dumps(payload)}")
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            return cls._parse_response(resp)
        except Exception as e:
            empty_image = cls._create_empty_image()
            return io.NodeOutput(empty_image,f'{{"success":false,"message":"The API request failed. Please check if the interface address and key are correct."}}')    
    # 将图像张量输入转换为与 Gemini API 兼容的格式。
    @classmethod
    def _create_image_parts(cls,image_input):
        image_parts: list[dict] = []
        for image_index in range(image_input.shape[0]):
            image_as_b64 = tensor_to_base64_string(
                image_input[image_index].unsqueeze(0)
            )
            inlineData ={
                "mimeType": "image/png",
                "data": image_as_b64
            }
            GeminiPart = {
                "inlineData": inlineData
            }
            image_parts.append(GeminiPart)
        return image_parts
    # 文生图模式
    @classmethod
    def _generate_images(cls,api_url,api_key,prompt,model,aspectRatio,imageSize,enableSearch,seed,timeout,proxies=None)-> io.NodeOutput:
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
        # 根据模型类型添加imageConfig - gemini-2.5-flash-image不支持imageSize参数
        if model == "gemini-2.5-flash-image":
            # 对于不支持imageSize的模型，只添加aspectRatio（如果非auto）
            if aspectRatio != "auto":
                payload["generationConfig"]["imageConfig"] = {
                    "aspectRatio": aspectRatio
                }
        elif model == "gemini-3-pro-image-preview":
            # 对于支持imageSize的模型，添加imageSize和aspectRatio
            image_config = {
                "imageSize": imageSize
            }
            if aspectRatio != "auto":
                image_config["aspectRatio"] = aspectRatio
            payload["generationConfig"]["imageConfig"] = image_config
            if enableSearch:
                payload["tools"] = [{"google_search": {}}]

        # print(f"正在请求Gemini文生图API: {api_url}")
        # print(f"请求载荷: {json.dumps(payload)}")
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            return cls._parse_response(resp)
        except Exception as e:
            empty_image = cls._create_empty_image()
            return io.NodeOutput(empty_image,f'{{"success":false,"message":"The API request failed. Please check if the interface address and key are correct."}}')
    # 解析response 返回内容
    @classmethod
    def _parse_response(cls,resp):
        # 检查HTTP状态码
        if resp.status_code != 200:
            empty_image = cls._create_empty_image()
            return (empty_image,f'{{"success":false,"message":"API request returns an error.status_code:{resp.status_code}.error_reason:{resp.text}"}}')
        # 检查返回内容是否为空
        if not resp.text.strip():
            empty_image = cls._create_empty_image()
            return (empty_image,f'{{"success":false,"message":"The API returns an empty content"}}')
        try:
            data = resp.json()
        except Exception as json_exception:
            # print(f"JSON解析失败：{json_exception}")
            empty_image = cls._create_empty_image()
            return (empty_image,f'{{"success":false,"message":"The API returned a JSON parsing failure"}}')
        # 解析响应数据
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            # 查找图像部分
            for part in parts:
                if "inlineData" in part:
                    inline_data = part["inlineData"]
                    mime_type = inline_data.get("mimeType", "")
                    image_date = inline_data.get("data", "")               
                    if mime_type.startswith("image/"):
                        comfyui_image = cls._image_data_to_comfyui_image(image_date)
                        # 解析usage信息
                        usageMetadata = data.get("usageMetadata", {})
                        tokens_usage = cls._format_tokens_usage(usageMetadata)
                        return (comfyui_image, tokens_usage)
        else:
            # print(f"未找到imag数据")
            empty_image = cls._create_empty_image()
            return (empty_image,f'{{"success":false,"message":"Imag data not found"}}')
    # 将返回的图像数据解析为 comfyui 格式的 image
    @classmethod
    def _image_data_to_comfyui_image(cls,image_date):
        try:
            image_bytes = base64.b64decode(image_date)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0) 
            return image_tensor
        except Exception as e:
            empty_image = cls._create_empty_image()
            return io.NodeOutput(empty_image,f'{{"success":false,"message":"Return image data parsing exception"}}')
    # 获取token用量
    @classmethod
    def _format_tokens_usage(cls,usageMetadata):
        if not usageMetadata:
            return ""
        total_tokens = usageMetadata.get('totalTokenCount', '-')
        return f'{{"success":true,"message":"total_tokens:{total_tokens}"}}'
    # 创建空图像
    @classmethod
    def _create_empty_image(cls):
        try:
            return torch.zeros(1, 512, 512, 3, dtype=torch.float32)
        except Exception as e:
            return None

# 设置 web 目录，该目录中的任何 .js 文件都将被前端加载为前端扩展
# WEB_DIRECTORY = "./somejs"


# 使用 router 添加自定义 API 路由
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/hello")
async def get_hello(request):
    return web.json_response("hello")
