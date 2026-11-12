"""Load the converted Drama 3 files (`tools/convert.py`) as ComfyUI-managed models."""
import contextlib
import json
import logging
import os

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from .codec import Drama3Codec
from .config import LINE_HEAD_IDS, LINE_HEAD_KEY, parse_config
from .model import Drama3Model, SecondGPU
from .offload import ExpertStore, HostEmbedding

TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish", "tokenizer.json")
# The LM head rows of the newline and <|audio_separator|> of the first release, which its converted files leave out: a
# dialogue's take says them between its lines.
LINE_HEAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish", "line_head.safetensors")
# What each card keeps besides its share of the layers: the first card holds the embedding table, the second the
# head, the tower and the caches.
FIRST_RESERVE = 3 * 1024 ** 3
SECOND_RESERVE = 4 * 1024 ** 3


class _OpsConfig:
    """`pick_operations` reads `.quant_config` off a model config; quantized files select `mixed_precision_ops`,
    which also carries the `MoEExperts` bank."""

    def __init__(self, quant_config):
        self.quant_config = quant_config


class Drama3:
    """A loaded Drama 3: its patchers, where each part runs, the tokenizer, and the (ids, rows) of the line head."""

    def __init__(self, model, patcher, line_head, second=None, second_patcher=None, experts=None):
        from tokenizers import Tokenizer
        self.model = model
        self.patcher = patcher
        self.second = second
        self.second_patcher = second_patcher
        self.experts = experts
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        self.line_head = (torch.tensor(LINE_HEAD_IDS), line_head)
        # how many separately generated lines are decoded in one batch (Drama 3 Speed Options): a line then hears its
        # speaker's latest take of the batch before, not the line before it
        self.lines_at_once = 1
        # the same seed gives the same take on every run: no experts are left to the CPU, whose arithmetic differs from
        # the kernels' in the last bits and whose share of a step depends on timing
        self.repeatable = False

    @property
    def config(self):
        return self.model.config

    def patchers(self):
        return [p for p in (self.patcher, self.second_patcher) if p is not None]

    def session(self, rows):
        """One generation's hold on the expert cache (experts in RAM): takes inside it share what earlier ones loaded.
        `rows`: the most rows, prompt and frames, a take inside it can reach."""
        if self.experts is None:
            return contextlib.nullcontext()
        return self.experts.session(self.patcher.load_device, self.patchers(), self.second is None, rows, not self.repeatable)

    def parts(self):
        """(layers, layer devices, final norm, audio head, tower): the head and tower run on the last device."""
        backbone = self.model.model
        first = self.patcher.load_device
        if self.second is None:
            layers = list(backbone.layers)
            return layers, [first] * len(layers), backbone.norm, self.model.audio_head, self.model.tower
        second = self.second_patcher.load_device
        layers = list(backbone.layers) + list(self.second.layers)
        devices = [first] * len(backbone.layers) + [second] * len(self.second.layers)
        return layers, devices, self.second.norm, self.second.audio_head, self.second.tower


def layer_bytes(layer):
    return sum(t.numel() * t.element_size() for t in layer.state_dict().values())


def auto_split(model, first_device, second_device):
    """The first layer index the second card runs: as many layers on the first card as its free memory allows,
    the rest on the second, which also holds the head and the tower."""
    sizes = [layer_bytes(layer) for layer in model.model.layers]
    tail = sum(t.numel() * t.element_size() for m in (model.audio_head, model.tower, model.model.norm) for t in m.state_dict().values())
    budget = comfy.model_management.get_total_memory(first_device) - FIRST_RESERVE - layer_bytes(model.model.embed_tokens)
    second_budget = comfy.model_management.get_total_memory(second_device) - SECOND_RESERVE - tail
    # balance the two cards' headroom
    best = None
    for first in range(1, len(sizes)):
        left = budget - sum(sizes[:first])
        right = second_budget - sum(sizes[first:])
        score = min(left, right)
        if best is None or score > best[0]:
            best = (score, first)
    if best[0] < 0:
        logging.warning("Drama 3: the model does not fit on both cards; ComfyUI will stream the rest of it")
    return best[1]


def load_drama3(path, second_device=None, split=0, experts_in_ram=False, expert_cache_gb=0.0):
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    if not metadata or "drama3_config" not in metadata:
        raise ValueError(f"{os.path.basename(path)} is not a converted Drama 3 file (tools/convert.py model)")
    config = parse_config(json.loads(metadata["drama3_config"]))
    config.fp8_experts = metadata.get("drama3_format") == "fp8"
    line_head = state.pop(LINE_HEAD_KEY) if LINE_HEAD_KEY in state else comfy.utils.load_torch_file(LINE_HEAD_PATH)["weight"]
    quant_config = comfy.utils.detect_layer_quantization(state, "")
    dtype = torch.bfloat16
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    operations = comfy.ops.pick_operations(dtype, dtype, load_device=load_device, model_config=_OpsConfig(quant_config))
    model = Drama3Model(config, device=offload_device, dtype=dtype, operations=operations)
    comfy.model_management.archive_model_dtypes(model)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)
    result = model.load_state_dict(state, strict=False, assign=patcher.is_dynamic())
    del state
    if result.unexpected_keys or result.missing_keys:
        raise ValueError(f"Drama 3 checkpoint keys: unexpected {result.unexpected_keys[:6]}, missing {result.missing_keys[:6]}")
    experts = None
    if experts_in_ram:
        experts = ExpertStore(int(expert_cache_gb * 1024 ** 3))
        for layer in model.model.layers:
            layer.mlp.experts = experts.add(layer.mlp.experts)
        model.model.embed_tokens = HostEmbedding(model.model.embed_tokens)
        model.evictable = []
    if second_device is None:
        return Drama3(model, patcher, line_head, experts=experts)

    first = split or auto_split(model, load_device, second_device)
    second = SecondGPU(model, first)
    second_patcher = comfy.model_patcher.CoreModelPatcher(second, load_device=second_device, offload_device=offload_device)
    logging.info("Drama 3: layers 0-%d on %s, layers %d-%d, head and tower on %s", first - 1, load_device, first,
                 config.text.num_hidden_layers - 1, second_device)
    return Drama3(model, patcher, line_head, second, second_patcher, experts)


class Codec:
    def __init__(self, model, patcher):
        self.model = model
        self.patcher = patcher


def load_codec(path):
    state = comfy.utils.load_torch_file(path)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.vae_offload_device()
    model = Drama3Codec(dtype=torch.float32, device=offload_device, operations=comfy.ops.manual_cast)
    comfy.model_management.archive_model_dtypes(model)
    model.load_state_dict(state, strict=True)
    return Codec(model, comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device))
