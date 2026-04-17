from comfy_api.latest import io

from .gemini_tts_node import GeminiTTS


class GeminiTTSMulti(io.ComfyNode):
    """
    这个节点使用谷歌 Gemini TTS API 生成人物对话语音
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_options = GeminiTTS._load_models_from_config()
        default_model = model_options[0]

        return io.Schema(
            node_id="YCYY_Gemini_TTS_Multi_API",
            display_name="Gemini TTS Multi API",
            category="YCYY/API/audio",
            inputs=[
                io.String.Input(
                    id="text",
                    default="## THE SCENE\n设置场景的背景信息，包括地点、氛围和环境细节，以确定基调和氛围。\n" \
                    "## DIRECTOR'S NOTES\n导演备注,仅定义对性能至关重要的内容，并注意不要过度指定。最常见的指令是风格、语速和口音，但模型不限于这些指令，也不要求使用这些指令。您可以随意添加自定义说明\n" \
                    "## TRANSCRIPT\n转写内容和音频标记,转写内容是模型将要朗读的确切字词。音频标记是指方括号中的字词，用于指示说话方式、音调变化或插话。多说话人名称需要与配置对应，示例如下：\n" \
                    "Speaker 1: I know right, I couldn't believe it. [whispers] She should have totally left at that point.\n" \
                    "Speaker 2: [cough] Well, [sighs] I guess it doesn't matter now.",
                    multiline=True,
                    tooltip="Conversation text. Include speaker labels that match the configured speaker names."
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
                io.AnyType.Input(
                    id="speaker_options",
                    tooltip="Speaker option array from Gemini Speaker Options or Gemini Batch Speakers Options"
                ),
                io.Combo.Input(
                    id="model",
                    options=model_options,
                    default=default_model
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
                io.Audio.Output(),
                io.String.Output()
            ],
            description="This node uses the Google Gemini TTS API to generate multi-speaker speech from text."
        )

    @classmethod
    def execute(cls, text, speaker_options, model, seed, config_options=None, proxy_options=None) -> io.NodeOutput:
        base_url, api_key, timeout = GeminiTTS._load_config_credentials(config_options)
        proxies = GeminiTTS._get_proxy_config(proxy_options)

        if not text:
            raise ValueError("text cannot be empty")

        normalized_speaker_options = cls._normalize_speaker_options(speaker_options)
        if not normalized_speaker_options:
            raise ValueError("speaker_options cannot be empty")

        api_url = base_url + "/" + model + ":generateContent"
        return cls._generate_speech(api_url, api_key, text, model, normalized_speaker_options, timeout, proxies)

    @classmethod
    def _normalize_speaker_options(cls, speaker_options):
        if not isinstance(speaker_options, list):
            raise ValueError("speaker_options must be a list")

        normalized = []
        seen_speakers = set()
        for item in speaker_options:
            if not isinstance(item, dict):
                raise ValueError("Each speaker option must be an object")

            speaker = str(item.get("speaker", "")).strip()
            voice_name = str(item.get("voiceName", "")).strip()

            if not speaker:
                raise ValueError("speaker cannot be empty")
            if not voice_name:
                raise ValueError("voiceName cannot be empty")
            if speaker in seen_speakers:
                raise ValueError(f"Duplicate speaker is not allowed: {speaker}")

            seen_speakers.add(speaker)
            normalized.append({
                "speaker": speaker,
                "voiceName": voice_name,
            })

        return normalized

    @classmethod
    def _generate_speech(cls, api_url, api_key, text, model, speaker_options, timeout, proxies=None) -> io.NodeOutput:
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

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
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {
                                "speaker": item["speaker"],
                                "voiceConfig": {
                                    "prebuiltVoiceConfig": {
                                        "voiceName": item["voiceName"]
                                    }
                                }
                            }
                            for item in speaker_options
                        ]
                    }
                }
            }
        }

        try:
            import requests
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
            return GeminiTTS._parse_response(resp)
        except Exception as e:
            return io.NodeOutput(None, f'{{"success":false,"message":"The API request failed. Please check if the interface address and key are correct. Error: {str(e)}"}}')
