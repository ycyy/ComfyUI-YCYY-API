from .gemini.gemini_image_node import *
from .gemini.gemini_image_preset_node import *
from .options.gemini_image_config_options_node import *
from .options.proxy_options_node import *
from .ollama.ollama_vlm_node import *
from .ollama.ollama_llm_node import *
from .options.ollama_llm_advanced_options_node import *
from typing_extensions import override

class APIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GeminiImage,
            GeminiImagePreset,
            GeminiImageConfigOptions,
            ProxyConfigOptions,
            OllamaVLM,
            OllamaLLM,
            OllamaLLMAdvanceOptions
        ]


async def comfy_entrypoint() -> APIExtension:
    return APIExtension()
