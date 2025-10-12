from .gemini_image_node import *
from .gemini_image_preset_node import *
from typing_extensions import override

class APIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            GeminiImage,
            GeminiImagePreset
        ]


async def comfy_entrypoint() -> APIExtension:
    return APIExtension()
