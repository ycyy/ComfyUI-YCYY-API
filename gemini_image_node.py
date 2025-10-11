import io
import os
import json
import base64
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .image_utils import tensor_to_base64_string

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
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
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
    def _load_config_credentials(cls):
        """
        从config.json中加载并验证API凭据
        返回 (base_url, api_key) 元组
        """
        config_path = os.path.join(os.path.dirname(__file__), "config.json")

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

            return base_url, api_key

        except Exception as e:
            raise ValueError(f"Config loading error: {str(e)}")

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
            display_name="YCYY Gemini Image API",
            category="YCYY/API/image",
            inputs=[
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional image(s) to use as context for the model"
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
    def execute(cls, prompt, model, aspectRatio, seed,images=None) -> io.NodeOutput:
        # 加载配置和凭据
        base_url, api_key = cls._load_config_credentials()
        if not prompt:
            raise ValueError("prompt cannot be empty")
        api_url = base_url+"/"+model+":generateContent"
        if images is not None:
            return cls._edit_images(api_url,api_key,prompt,aspectRatio,seed,images)
        else:
            return cls._generate_images(api_url,api_key,prompt,aspectRatio,seed)
    # 图生图模式
    @classmethod
    def _edit_images(cls,api_url,api_key,prompt,aspectRatio,seed,images)-> io.NodeOutput:
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
                # 如果aspectRatio不是auto，添加imageConfig
        if aspectRatio != "auto":
            payload["generationConfig"]["imageConfig"] = {
                "aspectRatio": aspectRatio
            }
        # print(f"正在请求Gemini文生图API: {api_url}")
        # print(f"请求载荷: {json.dumps(payload)}")
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            return cls._parse_response(resp)
        except Exception as e:
            empty_image = cls._create_empty_image()
            return io.NodeOutput(empty_image,"API请求失败，请检查接口地址和 KEY 是否正确")    
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
    def _generate_images(cls,api_url,api_key,prompt,aspectRatio,seed)-> io.NodeOutput:
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
        # 如果aspectRatio不是auto，添加imageConfig
        if aspectRatio != "auto":
            payload["generationConfig"]["imageConfig"] = {
                "aspectRatio": aspectRatio
            }
        # print(f"正在请求Gemini文生图API: {api_url}")
        # print(f"请求载荷: {json.dumps(payload)}")
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            return cls._parse_response(resp)
        except Exception as e:
            empty_image = cls._create_empty_image()
            return io.NodeOutput(empty_image,"API请求失败，请检查接口地址和 KEY 是否正确")
    # 解析response 返回内容
    @classmethod
    def _parse_response(cls,resp):
        # 检查HTTP状态码
        if resp.status_code != 200:
            empty_image = cls._create_empty_image()
            return (empty_image,f"API请求返回错误 (状态码: {resp.status_code}):(错误原因:{resp.text})")
        # 检查返回内容是否为空
        if not resp.text.strip():
            empty_image = cls._create_empty_image()
            return (empty_image,"API返回内容为空")
        try:
            data = resp.json()
        except Exception as json_exception:
            # print(f"JSON解析失败：{json_exception}")
            empty_image = cls._create_empty_image()
            return (empty_image,"API返回JSON解析失败")
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
            return (empty_image,"未找到imag数据")
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
            return io.NodeOutput(empty_image,"返回图像数据解析异常")
    # 获取token用量
    @classmethod
    def _format_tokens_usage(cls,usageMetadata):
        if not usageMetadata:
            return ""
        total_tokens = usageMetadata.get('totalTokenCount', '-')
        return f"{{\"success\":true,\"total_tokens\":{total_tokens}}}"
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


