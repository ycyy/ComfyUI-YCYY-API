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
from ..utils.config_utils import get_config_section, get_models_list

def _load_config_credentials():
    """
    从config.json中加载并验证API凭据
    返回 (base_url, api_key, timeout) 元组
    """
    try:
        ollama_llm_config = get_config_section('ollama-llm')
        # 获取并验证base_url
        if 'base_url' not in ollama_llm_config:
            raise ValueError("Missing 'base_url' in ollama-vlm section")
        base_url = ollama_llm_config['base_url'].strip() if isinstance(ollama_llm_config['base_url'], str) else str(ollama_llm_config['base_url']).strip()
        if not base_url:
            raise ValueError("base_url cannot be empty")

        # 对于Ollama，api_key是可选的
        api_key = ollama_llm_config.get('api_key', '')
        api_key = api_key.strip() if isinstance(api_key, str) else str(api_key).strip()

        # 获取timeout参数，默认值为120秒
        timeout = ollama_llm_config.get('timeout', 120)
        if isinstance(timeout, str):
            try:
                timeout = int(timeout)
            except ValueError:
                timeout = 120

        return base_url, api_key, timeout

    except Exception as e:
        raise ValueError(f"Failed to load Ollama VLM config section: {str(e)}")

class OllamaLLMAdvanceOptions(io.ComfyNode):
    """
    这个节点使用 Ollama LLM 模型进行对话
    """
    # 类级别的对话历史存储，按节点实例ID存储
    _conversation_history = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        # 从配置文件加载模型列表
        model_options = get_models_list("ollama-llm")
        default_model = model_options[0]
        return io.Schema(
            node_id="YCYY_Ollama_LLM_Advanced_Options",
            display_name="Ollama LLM Advanced Options",
            category="YCYY/API/utils",
            inputs=[
                io.Int.Input(
                    id="max_tokens",
                    default=4096,
                    min=1,
                    max=16384
                ),
                io.Float.Input(
                    id="temperature",
                    default=1,
                    min=0.0,
                    max=2.0,
                    step=0.01
                ),
                io.Float.Input(
                    id="top_p",
                    default=1,
                    min=0.0,
                    max=1.0,
                    step=0.01
                ),
                io.Boolean.Input(
                    id="enable_max_tokens",
                    default=True,
                    tooltip="Whether to enable max tokens"
                ), 
                io.Boolean.Input(
                    id="enable_temperature",
                    default=True,
                    tooltip="Whether to enable temperature"
                ),  
                io.Boolean.Input(
                    id="enable_top_p",
                    default=True,
                    tooltip="Whether to enable top_p"
                ),
                io.Boolean.Input(
                    id="enable_thinking",
                    default=False,
                    tooltip="Whether to enable thinking"
                )   
            ],
            outputs=[
                io.AnyType.Output(
                    id="advanced_options",
                    display_name="advanced_options",
                    tooltip="Optional configuration for the model."
                )
            ],
            description="This node uses the Ollama LLM model for conversation."
        )
    # 执行 OllamaLLM 节点
    @classmethod
    def execute(cls, max_tokens, temperature, top_p,enable_max_tokens,enable_temperature,enable_top_p,enable_thinking) -> io.NodeOutput:
        advanced_options = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "enable_max_tokens": enable_max_tokens,
            "enable_temperature": enable_temperature,
            "enable_top_p": enable_top_p,
            "enable_thinking": enable_thinking
        }
        return io.NodeOutput(advanced_options)

# 设置 web 目录，该目录中的任何 .js 文件都将被前端加载为前端扩展
# WEB_DIRECTORY = "./somejs"
