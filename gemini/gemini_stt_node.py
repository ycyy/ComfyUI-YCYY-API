import os
import json
import base64
import requests
import torch
import numpy as np
import io as python_io
import wave
from comfy_api.latest import io


class GeminiSTT(io.ComfyNode):
    """
    这个节点使用谷歌Gemini STT API 进行语音识别
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
                return ["gemini-2.5-flash", "gemini-2.5-pro"]

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'gemini-stt' in config and 'models' in config['gemini-stt']:
                models = config['gemini-stt']['models']
                if isinstance(models, list) and len(models) > 0:
                    return models

            return ["gemini-2.5-flash", "gemini-2.5-pro"]
        except Exception:
            return ["gemini-2.5-flash", "gemini-2.5-pro"]

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
        config_path = os.path.join(os.path.dirname(__file__), '..', "config.json")

        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 检查是否存在gemini-stt配置段
            if 'gemini-stt' not in config:
                raise ValueError("Missing 'gemini-stt' section in config file")

            stt_config = config['gemini-stt']

            # 获取并验证base_url
            if 'base_url' not in stt_config:
                raise ValueError("Missing 'base_url' in gemini-stt section")
            base_url = stt_config['base_url'].strip() if isinstance(stt_config['base_url'], str) else str(stt_config['base_url']).strip()
            if not base_url:
                raise ValueError("base_url cannot be empty")

            # 获取并验证api_key
            if 'api_key' not in stt_config:
                raise ValueError("Missing 'api_key' in gemini-stt section")
            api_key = stt_config['api_key'].strip() if isinstance(stt_config['api_key'], str) else str(stt_config['api_key']).strip()
            if not api_key:
                raise ValueError("api_key cannot be empty")

            # 获取timeout参数，默认值为120秒
            timeout = stt_config.get('timeout', 120)
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
            from ..utils.config_utils import get_config_section
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
        # 从配置文件加载模型列表
        model_options = cls._load_models_from_config()
        default_model = model_options[0]

        return io.Schema(
            node_id="YCYY_Gemini_STT_API",
            display_name="Gemini STT API",
            category="YCYY/API/audio",
            inputs=[
                io.Audio.Input(
                    id="audio",
                    tooltip="The audio to transcribe"
                ),
                io.String.Input(
                    id="prompt",
                    multiline=True,
                    default="",
                    tooltip="The prompt to guide the transcription. You can ask for specific formats or instructions."
                ),
                io.AnyType.Input(
                    id="config_options",
                    optional=True,
                    tooltip="Optional configuration override from YCYY Gemini STT Config Options"
                ),
                io.AnyType.Input(
                    id="proxy_options",
                    optional=True,
                    tooltip="Optional proxy configuration override from YCYY Proxy Config Options"
                ),
                io.Combo.Input(
                    id="model",
                    options=model_options,
                    default=default_model
                ),
            ],
            outputs=[
                io.String.Output(),    # Transcribed text
                io.String.Output()     # Metadata/usage info
            ],
            description="This node uses the Google Gemini STT API to transcribe speech to text."
        )

    @classmethod
    def execute(cls, audio, prompt, model, config_options=None, proxy_options=None) -> io.NodeOutput:
        # 加载配置和凭据，如果提供了config_options则使用覆盖配置
        base_url, api_key, timeout = cls._load_config_credentials(config_options)
        # 获取代理配置，如果提供了proxy_options则使用覆盖配置
        proxies = cls._get_proxy_config(proxy_options)

        if audio is None:
            raise ValueError("audio cannot be empty")

        if not prompt:
            prompt = "Please transcribe the audio."

        api_url = base_url + "/" + model + ":generateContent"

        return cls._transcribe_audio(api_url, api_key, audio, prompt, timeout, proxies)

    @classmethod
    def _transcribe_audio(cls, api_url, api_key, audio, prompt, timeout, proxies=None) -> io.NodeOutput:
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }
        # 将音频转换为base64
        audio_base64, mime_type = cls._audio_to_base64(audio)
        if not audio_base64:
            return io.NodeOutput("", '{"success":false,"message":"Failed to convert audio to base64"}')

        # 构建请求payload - 按照API示例格式
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": audio_base64
                            }
                        }
                    ]
                }
            ]
        }

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            return cls._parse_response(resp)
        except Exception as e:
            return io.NodeOutput("", f'{{"success":false,"message":"The API request failed. Please check if the interface address and key are correct. Error: {str(e)}"}}')

    @classmethod
    def _audio_to_base64(cls, audio):
        """
        将ComfyUI音频格式转换为base64编码的WAV文件
        audio格式: {'waveform': tensor, 'sample_rate': int}
        waveform shape: (batch, channels, samples)
        返回: (base64_string, mime_type)
        """
        try:
            # 提取音频数据
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            # 转换为numpy数组
            # waveform shape: (batch, channels, samples)
            # 取第一个batch，支持多声道
            audio_array = waveform[0].numpy()  # shape: (channels, samples)

            # 转置为 (samples, channels) 以符合WAV格式要求
            if audio_array.ndim == 2:
                audio_array = audio_array.T  # shape: (samples, channels)
                num_channels = audio_array.shape[1]
            else:
                # 单声道情况
                num_channels = 1
                audio_array = audio_array.reshape(-1, 1)

            # 将float32 [-1, 1] 转换为int16 PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)

            # 创建WAV文件到内存
            wav_buffer = python_io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            # 获取WAV文件字节数据
            wav_bytes = wav_buffer.getvalue()

            # 编码为base64
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')

            # 使用标准WAV mime type
            mime_type = "audio/wav"

            return audio_base64, mime_type

        except Exception as e:
            return None, None

    @classmethod
    def _parse_response(cls, resp):
        # 检查HTTP状态码
        if resp.status_code != 200:
            return ("", f'{{"success":false,"message":"API request returns an error.status_code:{resp.status_code}.error_reason:{resp.text}"}}')

        # 检查返回内容是否为空
        if not resp.text.strip():
            return ("", f'{{"success":false,"message":"The API returns an empty content"}}')

        try:
            data = resp.json()
        except Exception as json_exception:
            return ("", f'{{"success":false,"message":"The API returned a JSON parsing failure: {str(json_exception)}"}}')

        # 解析响应数据
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # 提取文本内容
            transcribed_text = ""
            for part in parts:
                if "text" in part:
                    transcribed_text += part["text"]

            if transcribed_text:
                # 解析usage信息
                usageMetadata = data.get("usageMetadata", {})
                tokens_usage = cls._format_tokens_usage(usageMetadata)

                return (transcribed_text, tokens_usage)

        # 未找到文本数据
        return ("", f'{{"success":false,"message":"Transcribed text not found"}}')

    @classmethod
    def _format_tokens_usage(cls, usageMetadata):
        """
        格式化token使用信息
        """
        if not usageMetadata:
            return ""
        total_tokens = usageMetadata.get('totalTokenCount', '-')
        prompt_tokens = usageMetadata.get('promptTokenCount', '-')
        candidates_tokens = usageMetadata.get('candidatesTokenCount', '-')
        return f'{{"success":true,"message":"total_tokens:{total_tokens}, prompt_tokens:{prompt_tokens}, candidates_tokens:{candidates_tokens}"}}'
