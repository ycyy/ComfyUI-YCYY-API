import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "YCYY_OpenAI_Text_API";
let apiMap = new Map();

async function loadApis() {
    try {
        const response = await api.fetchApi("/ycyy/openai/apis/all");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        apiMap = new Map((Array.isArray(data) ? data : []).map(item => [item["api-name"], item]));
    } catch (error) {
        console.error("[YCYY] Failed to load OpenAI API list:", error);
    }
}

function applyModels(node, apiName, keepModel = false) {
    const selected = apiMap.get(apiName);
    const modelWidget = node.widgets?.find(widget => widget.name === "model");
    if (!selected || !modelWidget) return;
    const models = Array.isArray(selected.models) ? selected.models : [];
    modelWidget.options.values = models;
    if (!keepModel || !models.includes(modelWidget.value)) modelWidget.value = models[0] ?? "";
    app.canvas?.draw(true, true);
}

app.registerExtension({
    name: "YCYY.OpenAI.Text",
    async setup() { await loadApis(); },
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeType.comfyClass !== NODE_CLASS) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            const apiWidget = this.widgets?.find(widget => widget.name === "api_name");
            if (apiWidget) {
                const originalCallback = apiWidget.callback;
                apiWidget.callback = value => {
                    applyModels(this, value);
                    // Update the dependent combo before the original callback
                    // runs, so stale model values are never validated against
                    // the newly selected API's list.
                    originalCallback?.call(this, value);
                };
                setTimeout(() => applyModels(this, apiWidget.value, true), 0);
            }
            return result;
        };
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            const apiWidget = this.widgets?.find(widget => widget.name === "api_name");
            if (apiWidget && apiMap.size && !apiMap.has(apiWidget.value)) {
                const fallback = apiMap.keys().next().value;
                console.warn(`[YCYY] API "${apiWidget.value}" no longer exists; using "${fallback}"`);
                apiWidget.value = fallback;
            }
            if (apiWidget) applyModels(this, apiWidget.value, true);
            return result;
        };
    },
});
