import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "YCYY_Gemini_Image_Preset";

let presetMap = new Map();

async function loadAllPresets(retries = 1) {
    try {
        const response = await api.fetchApi("/ycyy/gemini/images/presets/all");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        presetMap = new Map((Array.isArray(data) ? data : []).map(p => [p.title, p]));
    } catch (error) {
        console.error("Error fetching presets:", error);
        if (retries > 0) setTimeout(() => loadAllPresets(retries - 1), 2000);
    }
}

function applyPreset(node, descriptionWidget, promptWidget, title) {
    const preset = presetMap.get(title);
    if (!preset) return false;
    if (typeof preset.description === "string") descriptionWidget.value = preset.description;
    if (typeof preset.prompt === "string") promptWidget.value = preset.prompt;
    app.canvas?.draw(true, true);
    return true;
}

app.registerExtension({
    name: "YCYY.Gemini.Image.Preset",
    async setup() {
        await loadAllPresets();
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const presetWidget = this.widgets?.find(w => w.name === "preset");
            const descriptionWidget = this.widgets?.find(w => w.name === "description");
            const promptWidget = this.widgets?.find(w => w.name === "prompt");

            if (presetWidget && descriptionWidget && promptWidget) {
                const originalCallback = presetWidget.callback;
                const node = this;

                presetWidget.callback = (value) => {
                    originalCallback?.call(this, value);
                    if (!value || value === "None") return;
                    if (presetMap.size) {
                        applyPreset(node, descriptionWidget, promptWidget, value);
                    }
                };

                setTimeout(() => {
                    if (node.properties._ycyy_preset_filled) return;
                    if (!descriptionWidget.value && !promptWidget.value && presetWidget.value && presetWidget.value !== "None") {
                        if (applyPreset(node, descriptionWidget, promptWidget, presetWidget.value)) {
                            node.properties._ycyy_preset_filled = true;
                        }
                    }
                }, 0);
            }

            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);

            const presetWidget = this.widgets?.find(w => w.name === "preset");
            if (presetWidget) {
                const values = presetWidget.options?.values || [];
                if (presetWidget.value && presetWidget.value !== "None" && values.length && !values.includes(presetWidget.value)) {
                    const fallback = values.find(v => v !== "None") ?? values[0];
                    console.warn(`[YCYY] Preset "${presetWidget.value}" no longer exists, falling back to "${fallback}"`);
                    presetWidget.value = fallback;
                }
            }

            return r;
        };
    }
});
