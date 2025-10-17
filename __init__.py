from .gemini.gemini_image_node import *
from .gemini.gemini_image_preset_node import *
from .ollama.ollama_vlm_node import *
from .ollama.ollama_llm_node import *
from typing_extensions import override

class APIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GeminiImage,
            GeminiImagePreset,
            OllamaVLM,
            OllamaLLM
        ]


async def comfy_entrypoint() -> APIExtension:
    return APIExtension()
