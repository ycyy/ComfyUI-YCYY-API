import io
import os
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

# 公共函数，用于处理预设数据（带 mtime 缓存，文件变更自动重载）
_preset_cache = {"mtime": None, "data": []}

def load_preset_data():
    """加载预设数据从JSON配置文件（mtime变化时重新读取）"""
    config_path = os.path.join(os.path.dirname(__file__), "gemini_image_preset.json")
    try:
        mtime = os.path.getmtime(config_path)
        if _preset_cache["mtime"] != mtime:
            with open(config_path, 'r', encoding='utf-8') as f:
                _preset_cache["data"] = json.load(f)
            _preset_cache["mtime"] = mtime
        return _preset_cache["data"]
    except Exception as e:
        print(f"Error loading preset config: {e}")
        return _preset_cache["data"] or []

def get_preset_titles():
    """获取所有预设的标题列表"""
    preset_data = load_preset_data()
    if preset_data:
        return [preset['title'] for preset in preset_data]
    return ["None"]

class GeminiImagePreset(io.ComfyNode):
    """
    这个节点为Gemini Image API 提供预设
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
            返回一个包含该节点所有信息的模式（schema）。
            一些可用类型："Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo"。
            对于输出，应使用 "io.Model.Output"，对于输入，可以使用 "io.Model.Input"。
            类型可以是 "Combo" —— 这将是一个供选择的列表。
        """
        # 从配置文件加载预设
        preset_options = get_preset_titles()

        return io.Schema(
            node_id="YCYY_Gemini_Image_Preset",
            display_name="Gemini Image Preset",
            category="YCYY/API/utils",
            inputs=[
                io.Combo.Input(
                    id="preset",
                    options=preset_options,
                    default=preset_options[0] if preset_options and preset_options[0] != "None" else "None",
                    tooltip="Gemini image preset name"
                ),
                io.String.Input(
                    id="description",
                    multiline=True,
                    tooltip="Gemini image preset description"
                ),
                io.String.Input(
                    id="prompt",
                    multiline=True,
                    tooltip="Gemini image preset prompt"
                ),
            ],
            outputs=[
                io.String.Output()
            ],
            description="This node provides presets for the Gemini Image API."
        )
    # 执行 GeminiImagePreset 节点
    @classmethod
    def execute(cls, preset, description, prompt) -> io.NodeOutput:
        return (prompt,)

WEB_DIRECTORY = "./web/js"

# 使用 router 添加自定义 API 路由
from aiohttp import web
from server import PromptServer




@PromptServer.instance.routes.get("/ycyy/gemini/images/presets/all")
async def get_all_presets(request):
    """获取全部preset数据的API端点"""
    return web.json_response(load_preset_data())

