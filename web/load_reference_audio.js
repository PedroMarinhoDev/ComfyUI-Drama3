// Load Reference Audio gets the audio player ComfyUI gives its own Load Audio node (only core nodes get it), with the
// upload button under it as there: the button looks the player up when it is made. After a run the player plays the
// trimmed clip; picking another file plays that file again.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

app.registerExtension({
  name: "Drama3.LoadReferenceAudio",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Drama3LoadReferenceAudio") return;
    nodeData.input.required.audioUI = ["AUDIO_UI", {}];
    nodeData.input_order.required = ["audio", "audioUI", "upload", ...nodeData.input_order.required.filter(name => name !== "audio")];
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      const clip = output?.audio?.[0];
      const player = this.widgets?.find(widget => widget.name === "audioUI");
      if (clip && player) {
        player.element.src = api.apiURL(`/view?${new URLSearchParams({ filename: clip.filename, subfolder: clip.subfolder ?? "", type: clip.type })}`);
      }
    };
  },
});
