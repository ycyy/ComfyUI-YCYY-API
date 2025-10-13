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
from .image_utils import tensor_to_base64_string

# 公共函数，用于处理预设数据
def load_preset_data():
    """加载预设数据从JSON配置文件"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "gemini_image_preset.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading preset config: {e}")
        return []

def get_preset_titles():
    """获取所有预设的标题列表"""
    preset_data = load_preset_data()
    if preset_data:
        return [preset['title'] for preset in preset_data]
    return ["None"]

def get_preset_by_title(title):
    """根据title获取对应的预设数据"""
    preset_data = load_preset_data()
    for preset in preset_data:
        if preset.get('title') == title:
            return preset
    return None

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




@PromptServer.instance.routes.post("/ycyy/gemini/images/preset")
async def get_preset_data(request):
    """获取preset数据的API端点"""
    try:
        data = await request.json()
        preset_title = data.get('title')

        if not preset_title:
            return web.json_response({"error": "Missing preset title"}, status=400)

        preset_data = get_preset_by_title(preset_title)

        if preset_data:
            return web.json_response(preset_data)
        else:
            return web.json_response({"error": f"Preset '{preset_title}' not found"}, status=404)

    except Exception as e:
        return web.json_response({"error": f"Internal server error: {str(e)}"}, status=500)

