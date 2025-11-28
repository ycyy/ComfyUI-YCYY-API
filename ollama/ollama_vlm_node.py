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
from ..utils.config_utils import get_config_section,get_models_list


def _load_config_credentials(config_options=None):
    """
    从config.json中加载并验证API凭据，如果提供了config_options则优先使用
    返回 (base_url, api_key, timeout) 元组
    """
    # 如果提供了配置覆盖，则使用覆盖配置
    if config_options is not None:
        base_url = config_options.get('base_url', '').strip()
        api_key = config_options.get('api_key', '').strip()
        timeout = config_options.get('timeout', 120)

        # 如果覆盖配置中有有效的 base_url，则直接返回
        if base_url:
            return base_url, api_key, timeout

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
        raise ValueError(f"Failed to load Ollama VLM config section: {str(e)}")

def _get_proxy_config(proxy_options=None):
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
        model_options = get_models_list("ollama-vlm")
        default_model = model_options[0]
        return io.Schema(
            node_id="YCYY_Ollama_VLM_API",
            display_name="Ollama VLM API",
            category="YCYY/API/text",
            inputs=[
                io.AnyType.Input(
                    id="config_options",
                    optional=True,
                    tooltip="Optional configuration override from YCYY Ollama Config Options"
                ),
                io.AnyType.Input(
                    id="proxy_options",
                    optional=True,
                    tooltip="Optional proxy configuration override from YCYY Proxy Config Options"
                ),
                io.Image.Input(
                    "images",
                    tooltip="Image used for analysis"
                ),
                io.String.Input(
                    id="system_prompt",
                    multiline=True,
                ),
                io.String.Input(
                    id="user_prompt",
                    multiline=True,
                ),
                io.Combo.Input(
                    id="model",
                    options=model_options,
                    default=default_model
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
    # 执行 OllamaVLM 节点
    @classmethod
    def execute(cls,images, system_prompt, user_prompt, model, config_options=None, proxy_options=None) -> io.NodeOutput:
        if not user_prompt:
            raise ValueError("User prompt cannot be empty")

        base_url, api_key, timeout = _load_config_credentials(config_options)
        proxies = _get_proxy_config(proxy_options)
        api_url = base_url
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

        # 构建用户消息内容，支持多个图片
        content = [
            {
                "type": "text",
                "text": user_prompt
            }
        ]

        # 处理多个图片
        if images is not None:
            for image_index in range(images.shape[0]):
                image_base64 = tensor_to_base64_string(images[image_index].unsqueeze(0))
                content.append({
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{image_base64}"
                })

        user_message = {
            "role": "user",
            "content": content
        }
        payload["messages"].append(user_message)
        try:
            if api_key:
                headers = {
                    "Authorization": "Bearer "+api_key
                }
                resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            else:
                resp = requests.post(api_url, json=payload, timeout=timeout, proxies=proxies)
            print(resp)
            return cls._parse_response(resp)
        except Exception as e:
            raise ValueError(f'The API request failed:{e}')
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
        # 解析响应数据 - OpenAI兼容接口格式
        if "choices" not in data:
            raise ValueError(f'Missing "choices" field in API response')

        choices = data["choices"]
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise ValueError(f'Empty or invalid "choices" array in API response')

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError(f'Invalid choice format in API response')

        message = first_choice.get("message", {})
        if not message:
            raise ValueError(f'Missing "message" field in API response')

        content = message.get("content", "")
        if not content:
            raise ValueError(f'Empty content in API response')

        return io.NodeOutput(content)

# 设置 web 目录，该目录中的任何 .js 文件都将被前端加载为前端扩展
# WEB_DIRECTORY = "./somejs"
