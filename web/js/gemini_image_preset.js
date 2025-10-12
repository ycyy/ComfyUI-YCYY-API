import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "YCYY.Gemini.Image.Preset",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "YCYY_Gemini_Image_Preset") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);

                const presetWidget = this.widgets?.find(w => w.name === "preset");
                const descriptionWidget = this.widgets?.find(w => w.name === "description");
                const promptWidget = this.widgets?.find(w => w.name === "prompt");

                if (presetWidget && descriptionWidget && promptWidget) {
                    const originalCallback = presetWidget.callback;

                    presetWidget.callback = async (value) => {
                        originalCallback?.call(this, value);

                        if (!value || value === "None") return;

                        try {
                            const response = await api.fetchApi("/ycyy/gemini/images/preset", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ title: value })
                            });

                            if (response.ok) {
                                const presetData = await response.json();
                                if (presetData?.description && presetData?.prompt) {
                                    descriptionWidget.value = presetData.description;
                                    promptWidget.value = presetData.prompt;
                                    this.graph?.canvas?.draw(true, true);
                                }
                            }
                        } catch (error) {
                            console.error("Error fetching preset data:", error);
                        }
                    };

                    if (presetWidget.value && presetWidget.value !== "None") {
                        setTimeout(() => presetWidget.callback(presetWidget.value), 50);
                    }
                }

                return r;
            };
        }
    }
});