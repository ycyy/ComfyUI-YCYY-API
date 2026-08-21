from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .gemini.gemini_image_node import *
from .gemini.gemini_image_preset_node import *
from .gemini.gemini_tts_node import *
from .gemini.gemini_tts_multi_node import *
from .gemini.gemini_stt_node import *
from .ollama.ollama_vlm_node import *
from .ollama.ollama_llm_node import *
from .options.ollama_llm_advanced_options_node import *
from .modelscope.modelscope_image_node import *
from .modelscope.modelscope_image_edit_node import *
from .options.config_options_node import *
from .options.gemini_speaker_options_node import *
from .options.gemini_batch_speakers_options_node import *
from .options.proxy_options_node import *
from .images.image_compare import ImageCompare

WEB_DIRECTORY = "./web/js"

class APIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GeminiImage,          
            GeminiTTS,
            GeminiTTSMulti,
            GeminiSTT,
            OllamaLLM,
            OllamaVLM,
            ModelScopeImage,
            ModelScopeImageEdit,
            ConfigOptions,
            ProxyOptions,
            GeminiImagePreset,
            GeminiSpeakerOptions,
            GeminiBatchSpeakersOptions,        
            OllamaLLMAdvanceOptions, 
            ImageCompare,
        ]


async def comfy_entrypoint() -> APIExtension:
    return APIExtension()
