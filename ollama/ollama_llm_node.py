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

class OllamaLLM(io.ComfyNode):
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
            node_id="YCYY_Ollama_LLM_API",
            display_name="Ollama LLM API",
            category="YCYY/API/text",
            inputs=[
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
                ),
                io.Boolean.Input(
                    id="persist_context",
                    default=True,
                    tooltip="Persist chat context between calls (multi-turn conversation)"
                )
                
            ],
            outputs=[
                io.String.Output(
                    id="Result",
                    display_name="Result",
                    tooltip="Return result"
                ),
                io.String.Output(
                    id="conversation",
                    display_name="Conversation",
                    tooltip="All historical conversation"
                )
            ],
            description="This node uses the Ollama LLM model for conversation."
        )
    # 执行 OllamaLLM 节点
    @classmethod
    def execute(cls, system_prompt, user_prompt, model, persist_context) -> io.NodeOutput:
        if not user_prompt:
            raise ValueError("User prompt cannot be empty")

        base_url, api_key, timeout = _load_config_credentials()
        api_url = base_url

        # 生成会话标识符（基于模型和系统提示词）
        session_key = f"{model}_{hash(system_prompt) if system_prompt else 'no_system'}"

        # 根据persist_context决定是否使用历史消息
        if persist_context:
            # 如果会话不存在，初始化历史记录
            if session_key not in cls._conversation_history:
                cls._conversation_history[session_key] = []
                # 如果有系统提示词，添加到历史记录开头
                if system_prompt:
                    cls._conversation_history[session_key].append({
                        "role": "system",
                        "content": system_prompt
                    })

            # 添加当前用户消息到历史
            cls._conversation_history[session_key].append({
                "role": "user",
                "content": user_prompt
            })

            # 使用完整的历史消息
            messages = cls._conversation_history[session_key].copy()
        else:
            # 不持久化上下文，清空历史并只使用当前消息
            if session_key in cls._conversation_history:
                del cls._conversation_history[session_key]

            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": user_prompt
            })

        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }

        try:
            if api_key:
                headers = {
                    "Authorization": "Bearer "+api_key
                }
                resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            else:
                resp = requests.post(api_url, json=payload, timeout=timeout)
            print(resp)
            return cls._parse_response(resp, persist_context, session_key)
        except Exception as e:
            raise ValueError(f'The API request failed:'+{e})
    # 解析response 返回内容
    @classmethod
    def _parse_response(cls, resp, persist_context, session_key):
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

        # 如果启用了上下文持久化，将助手的回复添加到历史记录
        if persist_context and session_key in cls._conversation_history:
            cls._conversation_history[session_key].append({
                "role": "assistant",
                "content": content
            })

        # 获取历史对话记录并转换为JSON字符串
        history_conversation = ""
        if persist_context and session_key in cls._conversation_history:
            history_conversation = json.dumps(cls._conversation_history[session_key], ensure_ascii=False)
        else:
            history_conversation = "[]"

        # 返回当前内容和历史对话JSON字符串
        return io.NodeOutput(content, history_conversation)

# 设置 web 目录，该目录中的任何 .js 文件都将被前端加载为前端扩展
# WEB_DIRECTORY = "./somejs"
