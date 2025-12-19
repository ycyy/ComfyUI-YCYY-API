import os
import json
import base64
import requests
import torch
import numpy as np
from comfy_api.latest import io


class GeminiTTS(io.ComfyNode):
    """
    这个节点使用谷歌Gemini TTS API 生成语音
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
                return ["gemini-2.5-flash-preview-tts"]

            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'gemini-tts' in config and 'models' in config['gemini-tts']:
                models = config['gemini-tts']['models']
                if isinstance(models, list) and len(models) > 0:
                    return models

            return ["gemini-2.5-flash-preview-tts"]
        except Exception:
            return ["gemini-2.5-flash-preview-tts"]

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

            # 检查是否存在gemini-tts配置段
            if 'gemini-tts' not in config:
                raise ValueError("Missing 'gemini-tts' section in config file")

            tts_config = config['gemini-tts']

            # 获取并验证base_url
            if 'base_url' not in tts_config:
                raise ValueError("Missing 'base_url' in gemini-tts section")
            base_url = tts_config['base_url'].strip() if isinstance(tts_config['base_url'], str) else str(tts_config['base_url']).strip()
            if not base_url:
                raise ValueError("base_url cannot be empty")

            # 获取并验证api_key
            if 'api_key' not in tts_config:
                raise ValueError("Missing 'api_key' in gemini-tts section")
            api_key = tts_config['api_key'].strip() if isinstance(tts_config['api_key'], str) else str(tts_config['api_key']).strip()
            if not api_key:
                raise ValueError("api_key cannot be empty")

            # 获取timeout参数，默认值为120秒
            timeout = tts_config.get('timeout', 120)
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
            node_id="YCYY_Gemini_TTS_API",
            display_name="Gemini TTS API",
            category="YCYY/API/audio",
            inputs=[
                io.String.Input(
                    id="text",
                    multiline=True,
                    tooltip="The text to convert to speech"
                ),
                io.AnyType.Input(
                    id="config_options",
                    optional=True,
                    tooltip="Optional configuration override from YCYY Gemini TTS Config Options"
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
                io.Combo.Input(
                    id="voiceName",
                    options=[
                        "Zephyr",
                        "Puck",
                        "Charon",
                        "Kore",
                        "Fenrir",
                        "Leda",
                        "Orus",
                        "Aoede",
                        "Callirrhoe",
                        "Autonoe",
                        "Enceladus",
                        "Iapetus",
                        "Umbriel",
                        "Algieba",
                        "Despina",
                        "Erinome",
                        "Algenib",
                        "Rasalgethi",
                        "Laomedeia",
                        "Achernar",
                        "Alnilam",
                        "Schedar",
                        "Gacrux",
                        "Pulcherrima",
                        "Achird",
                        "Zubenelgenubi",
                        "Vindemiatrix",
                        "Sadachbia",
                        "Sadaltager",
                        "Sulafat"
                    ],
                    default="Zephyr",
                    tooltip="The voice to use for speech synthesis"
                ),
            ],
            outputs=[
                io.Audio.Output(),  # Audio data as bytes/tensor
                io.String.Output()    # Metadata/usage info
            ],
            description="This node uses the Google Gemini TTS API to generate speech from text."
        )

    @classmethod
    def execute(cls, text, model, voiceName, config_options=None, proxy_options=None) -> io.NodeOutput:
        # 加载配置和凭据，如果提供了config_options则使用覆盖配置
        base_url, api_key, timeout = cls._load_config_credentials(config_options)
        # 获取代理配置，如果提供了proxy_options则使用覆盖配置
        proxies = cls._get_proxy_config(proxy_options)

        if not text:
            raise ValueError("text cannot be empty")

        api_url = base_url + "/" + model + ":generateContent"

        return cls._generate_speech(api_url, api_key, text, model, voiceName, timeout, proxies)

    @classmethod
    def _generate_speech(cls, api_url, api_key, text, model, voiceName, timeout, proxies=None) -> io.NodeOutput:
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

        # 构建请求payload - 严格按照API示例格式
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": text
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voiceName
                        }
                    }
                }
            }
        }

        # print(f"正在请求Gemini TTS API: {api_url}")
        # print(f"请求载荷: {json.dumps(payload)}")

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            return cls._parse_response(resp)
        except Exception as e:
            return io.NodeOutput(None, f'{{"success":false,"message":"The API request failed. Please check if the interface address and key are correct. Error: {str(e)}"}}')

    @classmethod
    def _parse_response(cls, resp):
        # 检查HTTP状态码
        if resp.status_code != 200:
            return (None, f'{{"success":false,"message":"API request returns an error.status_code:{resp.status_code}.error_reason:{resp.text}"}}')

        # 检查返回内容是否为空
        if not resp.text.strip():
            return (None, f'{{"success":false,"message":"The API returns an empty content"}}')

        try:
            data = resp.json()
        except Exception as json_exception:
            return (None, f'{{"success":false","message":"The API returned a JSON parsing failure: {str(json_exception)}"}}')

        # 解析响应数据
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # 查找音频部分
            for part in parts:
                if "inlineData" in part:
                    inline_data = part["inlineData"]
                    mime_type = inline_data.get("mimeType", "")
                    audio_data = inline_data.get("data", "")

                    if mime_type.startswith("audio/"):
                        # 解析 mime_type 获取音频参数
                        audio_params = cls._parse_mime_type(mime_type)

                        # 解析音频数据
                        audio_result = cls._process_audio_data(audio_data, audio_params)

                        # 解析usage信息
                        usageMetadata = data.get("usageMetadata", {})
                        tokens_usage = cls._format_tokens_usage(usageMetadata)

                        return (audio_result, tokens_usage)

        # 未找到音频数据
        return (None, f'{{"success":false,"message":"Audio data not found"}}')

    @classmethod
    def _parse_mime_type(cls, mime_type):
        """
        解析 mime_type 字符串，提取音频参数
        例如: "audio/L16;codec=pcm;rate=24000"
        返回包含采样率、编码格式等参数的字典
        """
        params = {
            'format': 'unknown',
            'codec': 'pcm',
            'sample_rate': 24000,  # 默认采样率
            'bits_per_sample': 16  # 默认位深
        }

        try:
            # 分割 mime_type
            parts = mime_type.split(';')
            if parts:
                # 第一部分是格式类型，例如 "audio/L16"
                params['format'] = parts[0].strip()

                # 解析参数
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip()

                        if key == 'rate':
                            try:
                                params['sample_rate'] = int(value)
                            except ValueError:
                                pass
                        elif key == 'codec':
                            params['codec'] = value.lower()

                # 根据格式推断位深
                if 'L16' in params['format'] or 'l16' in params['format']:
                    params['bits_per_sample'] = 16
                elif 'L8' in params['format'] or 'l8' in params['format']:
                    params['bits_per_sample'] = 8
                elif 'L24' in params['format'] or 'l24' in params['format']:
                    params['bits_per_sample'] = 24

        except Exception:
            pass

        return params

    @classmethod
    def _process_audio_data(cls, audio_base64, audio_params):
        """
        处理音频数据，根据格式要求进行转换
        返回符合 ComfyUI 要求的音频数据格式
        """
        try:
            # 解码base64音频数据
            audio_bytes = base64.b64decode(audio_base64)

            # 根据参数转换为音频 tensor
            return cls._bytes_to_audio_tensor(audio_bytes, audio_params)

        except Exception as e:
            return None

    @classmethod
    def _bytes_to_audio_tensor(cls, audio_bytes, audio_params):
        """
        将音频字节数据转换为 ComfyUI 要求的音频格式
        返回字典: {'waveform': tensor, 'sample_rate': int}
        waveform shape: (batch, channels, samples)
        """
        try:
            # 根据位深确定数据类型
            bits_per_sample = audio_params.get('bits_per_sample', 16)
            sample_rate = audio_params.get('sample_rate', 24000)

            # 根据位深选择合适的 numpy 类型和处理方式
            if bits_per_sample == 16:
                # 16位有符号整数，小端序
                dtype = np.dtype('<i2')  # little-endian int16
                max_value = 32768.0
                audio_array = np.frombuffer(audio_bytes, dtype=dtype)
                # 转换为float32并归一化到[-1, 1]
                audio_float = audio_array.astype(np.float32) / max_value

            elif bits_per_sample == 8:
                # 8位PCM通常是无符号整数 (0-255)
                dtype = np.uint8
                audio_array = np.frombuffer(audio_bytes, dtype=dtype)
                # 转换为float32并归一化到[-1, 1]
                # 先转换到 [0, 1]，然后映射到 [-1, 1]
                audio_float = (audio_array.astype(np.float32) / 127.5) - 1.0

            elif bits_per_sample == 24:
                # 24位PCM: 每个样本3字节，小端序
                # 将3字节转换为int32，然后处理
                num_samples = len(audio_bytes) // 3
                audio_array = np.zeros(num_samples, dtype=np.int32)

                for i in range(num_samples):
                    # 读取3字节（小端序）并扩展为4字节int32
                    # 保持符号位
                    byte1 = audio_bytes[i * 3]
                    byte2 = audio_bytes[i * 3 + 1]
                    byte3 = audio_bytes[i * 3 + 2]

                    # 组合成24位值
                    value = byte1 | (byte2 << 8) | (byte3 << 16)

                    # 处理符号扩展（如果最高位是1，说明是负数）
                    if value & 0x800000:
                        value |= 0xFF000000  # 符号扩展到32位

                    audio_array[i] = np.int32(value)

                # 归一化：24位有符号整数范围是 -8388608 到 8388607
                max_value = 8388608.0
                audio_float = audio_array.astype(np.float32) / max_value

            else:
                # 默认使用16位处理
                dtype = np.dtype('<i2')  # little-endian int16
                max_value = 32768.0
                audio_array = np.frombuffer(audio_bytes, dtype=dtype)
                audio_float = audio_array.astype(np.float32) / max_value

            # 检查是否为空
            if audio_float.size == 0:
                return None

            # 转换为tensor
            # ComfyUI 期望的格式: (batch, channels, samples)
            # 单声道音频: (1, 1, samples)
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).unsqueeze(0)

            # 返回 ComfyUI 标准音频格式
            return {
                'waveform': audio_tensor,
                'sample_rate': sample_rate
            }
        except Exception as e:
            return None

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
