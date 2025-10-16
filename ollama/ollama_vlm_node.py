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


def _load_ollama_vlm_models():
    """
    从config.json中加载ollama-vlm配置并获取模型列表
    """
    try:
        ollama_vlm_config = get_config_section('ollama-vlm')

        # 验证配置是否存在
        if not ollama_vlm_config:
            raise ValueError("Missing 'ollama-vlm' section in config file")

        # 直接获取models列表
        if 'models' not in ollama_vlm_config:
            raise ValueError("Missing 'models' in ollama-vlm section")

        models = ollama_vlm_config['models']

        # 验证models是否为列表且不为空
        if not isinstance(models, list):
            raise ValueError("'models' must be a list")

        if not models:
            raise ValueError("'models' list cannot be empty")
        return models
    except Exception as e:
        raise ValueError(f"Failed to load Ollama VLM models: {str(e)}")

def _load_config_credentials():
    """
    从config.json中加载并验证API凭据
    返回 (base_url, api_key, timeout) 元组
    """
    try:
        ollama_vlm_config = get_config_section('ollama-vlm')
        # 获取并验证base_url
        if 'base_url' not in ollama_vlm_config:
            raise ValueError("Missing 'base_url' in ollama-vlm section")
        base_url = ollama_vlm_config['base_url'].strip() if isinstance(ollama_vlm_config['base_url'], str) else str(ollama_vlm_config['base_url']).strip()
        if not base_url:
            raise ValueError("base_url cannot be empty")

        # 对于Ollama，api_key是可选的
        api_key = ollama_vlm_config.get('api_key', '')
        api_key = api_key.strip() if isinstance(api_key, str) else str(api_key).strip()

        # 获取timeout参数，默认值为120秒
        timeout = ollama_vlm_config.get('timeout', 120)
        if isinstance(timeout, str):
            try:
                timeout = int(timeout)
            except ValueError:
                timeout = 120

        return base_url, api_key, timeout

    except Exception as e:
        raise ValueError(f"Failed to load Ollama VLM config section: {str(e)}")

class OllamaVLM(io.ComfyNode):
    """
    这个节点使用 Ollama VLM 模型进行图片推理分析
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        """
            返回一个包含该节点所有信息的模式（schema）。
            一些可用类型："Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo"。
            对于输出，应使用 "io.Model.Output"，对于输入，可以使用 "io.Model.Input"。
            类型可以是 "Combo" —— 这将是一个供选择的列表。
        """
        # 从配置文件加载模型列表
        model_options = _load_ollama_vlm_models()
        default_model = model_options[0]
        return io.Schema(
            node_id="YCYY_Ollama_VLM_API",
            display_name="Ollama VLM API",
            category="YCYY/API/text",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Image used for analysis"
                ),
                io.Combo.Input(
                    id="model",
                    options=model_options,
                    default=default_model
                ),
                io.String.Input(
                    id="system_prompt",
                    multiline=True,
                ),
                io.String.Input(
                    id="user_prompt",
                    multiline=True,
                )
            ],
            outputs=[
                io.String.Output()
            ],
            description="This node uses the Ollama VLM model for image reasoning and analysis."
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
    def execute(cls,image, system_prompt, user_prompt, model) -> io.NodeOutput:
        if not user_prompt:
            raise ValueError("User prompt cannot be empty")

        base_url, api_key, timeout = _load_config_credentials()
        api_url = base_url+"/api/chat"
        payload = {
            "model": model,
            "messages": [],
            "stream": False
        }
        if system_prompt:
            system_message = {
                "role": "system",
                "content": system_prompt
            }
            payload["messages"].append(system_message)
        image_base64 = tensor_to_base64_string(image)
        user_message  ={
            "role": "user",
            "content": user_prompt,
            "images":[image_base64]
        }
        payload["messages"].append(user_message)
        try:
            if api_key:
                headers = {
                    "Authorization": "Bearer "+api_key
                }
                resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            else:
                resp = requests.post(api_url, json=payload, timeout=timeout)
            print(resp)
            return cls._parse_response(resp)
        except Exception as e:
            raise ValueError(f'The API request failed:'+{e})
    # 解析response 返回内容
    @classmethod
    def _parse_response(cls,resp):
        # 检查HTTP状态码
        if resp.status_code != 200:
            raise ValueError(f'API request returns an error.status_code:{resp.status_code}.error_reason:{resp.text}')
        # 检查返回内容是否为空
        if not resp.text.strip():
            raise ValueError(f'The API returns an empty content')
        try:
            data = resp.json()
        except Exception as json_exception:
            # print(f"JSON解析失败：{json_exception}")
            raise ValueError(f'The API returned a JSON parsing failure')
        # 解析响应数据
        if "message" in data and data["message"]:
            message = data.get("message", {})
            content = message.get("content","")
            return io.NodeOutput(content)
        else:
            raise ValueError(f'Content data not found')

# 设置 web 目录，该目录中的任何 .js 文件都将被前端加载为前端扩展
# WEB_DIRECTORY = "./somejs"
