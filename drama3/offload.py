"""Routed experts held in host memory instead of VRAM, for cards that cannot hold all 32 B of them.

A decode step uses 8 of a layer's 256 experts, so only those reach the GPU, and a cache in the VRAM left free keeps
the most used ones. With int8 experts, the CPU runs most of the missed ones while the GPU runs the cached ones and the
rest, and the most used of the CPU's are copied into the cache between steps. The cache lives for one generation, every take of it. A prompt uses nearly every expert: its pass
runs the cached ones from the cache and the others from host memory, keeping the most used of them while slots are free.

With int8 or FP8 experts the decode kernels (`kernels.py`) keep the cache's table on the GPU and read a missed expert
straight from host memory, so a decode step never waits on the host and runs as CUDA graphs. A prompt's pass runs each
layer's experts as one grouped kernel, and int8 experts with few rows run on the CPU meanwhile: it reads host memory
faster than PCIe carries it. Other formats look the cache up on the host and copy missed experts in.
"""
import concurrent.futures
import contextlib
import dataclasses
import itertools
import logging
import time

import torch
import torch.nn as nn

import comfy.model_management
import comfy.ops
from comfy.quant_ops import QuantizedTensor
from . import kernels

# VRAM a generation needs besides the model and the cache: a prompt's pass and the KV caches grow with the rows
# (0.29 MiB per row on prompts of 200 to 3,900 rows), on top of the kernels' and graphs' own memory
RESERVE_BASE = 512 * 1024 ** 2
RESERVE_PER_ROW = 0.3 * 1024 ** 2
# comfy_aimdo reads the device's free memory at most every 2 s: after this long its next reading is a new one
EVICTION_PAUSE = 2.0
FP8_BLOCK = 128
# a prompt's pass splits its uncached int8 experts between the CPU and the GPU by these costs (an i9-14900K and PCIe
# 4.0 x8), which each pass then scales by what it measured
CPU_EXPERT_US = 60
CPU_ROW_US = 13
PCIE_BYTES_PER_US = 12e3
# passes (prompts or decode steps) over which an expert's use count halves: the cache keeps the most used experts,
# which on a recorded take hit 85% of reads against 82% for least recently used
HALF_LIFE = 128
# the decode kernels' and the CPU side's limit on experts per row (the model routes 8)
MAX_EXPERTS_PER_ROW = 8
# slots a decode step may reserve for experts the CPU ran (the most used first), copied in between steps; fewer
# while the copies lag
RESERVE_START, RESERVE_MIN, RESERVE_MAX = 32, 8, 256
# ...and only for experts used this many times (decayed): one-off experts would push used ones out
ADMIT = 6.0
# how long a decode layer waits for the host's experts before running them itself (a busy or late host)
AWAIT_US = 5000


def pinned(tensor):
    if comfy.model_management.DISABLE_PINNED_MEMORY:
        return tensor.contiguous()
    out = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    return out.copy_(tensor)


def tensor_fields(weight):
    params = weight._params
    return {f.name: getattr(params, f.name) for f in dataclasses.fields(params) if torch.is_tensor(getattr(params, f.name))}


def per_row_fields(weight):
    """The bank's tensors that hold one entry per weight row: these are sliced or gathered with the rows."""
    rows = weight._qdata.shape[0]
    return {"_qdata": weight._qdata, **{n: v for n, v in tensor_fields(weight).items() if v.ndim >= 1 and v.shape[0] == rows}}


def kernel_bank(weight):
    """(kind, fields) of a flat quantized bank [E * out, ...] the decode kernels read, or None: "int8" (ConvRot) with
    {"_qdata", "scale"}; "w4" or "w6" (W4A8 with its codebook, uniform W6A8) with {"_qdata", "scale", "s_channel"}."""
    if not isinstance(weight, QuantizedTensor) or weight._qdata.ndim != 2:
        return None
    params, fields = weight._params, per_row_fields(weight)
    if params.orig_shape[-1] > 2048:
        return None
    if weight._layout_cls == "TensorWiseINT8Layout":
        if getattr(params, "convrot", False) and params.convrot_groupsize == 256 and set(fields) == {"_qdata", "scale"}:
            return "int8", fields
        return None
    if (weight._layout_cls == "AsymW4A8Int8Layout" and params.group_size == 16 and params.convrot_groupsize == 256
            and params.correction is None and not params.transposed and set(fields) == {"_qdata", "scale", "s_channel"}):
        bits = weight._qdata.shape[1] * 8 // params.orig_shape[-1]
        # 4 bits read a codebook, 6 bits are uniform
        if (bits, params.codebook is not None) in ((4, True), (6, False)):
            return f"w{bits}", fields
    return None


def quantized_gemv(kind, x, ids, plan, took_over, late, slot_fields, host_fields, rows, codebook):
    """[k, rows]: a decode row's experts in an int8, W4A8 or W6A8 bank against x (float32, rotated; see
    kernels.gemv_int8 / gemv_codes)."""
    if kind == "int8":
        return kernels.gemv_int8(x, ids, plan, took_over, late, slot_fields, host_fields, rows)
    return kernels.gemv_codes(x, ids, plan, slot_fields, host_fields, codebook, int(kind[1]), rows)


def quantized_grouped(kind, x, groups, slot_fields, host_fields, rows, codebook):
    """quantized_gemv for a prompt's rows (see kernels.grouped_int8 / grouped_codes)."""
    if kind == "int8":
        return kernels.grouped_int8(x, groups, slot_fields, host_fields, rows)
    return kernels.grouped_codes(x, groups, slot_fields, host_fields, codebook, int(kind[1]), rows)


def resident_decode(x, ids, weights, banks, rows, kind, codebooks):
    """One row x [1, H] through experts `ids` [k] of banks held whole on the GPU (fields laid out like the slot cache's,
    expert e in slot e, `rows` per expert; a codebook per bank for W4A8) with the decode kernels: each expert read in
    place, nothing waiting on the host."""
    (gate_up, down), (gate_up_rows, down_rows) = banks, rows
    plan = torch.stack((ids.to(torch.int32), torch.zeros_like(ids, dtype=torch.int32)), dim=-1)
    if kind == "fp8":
        gate, up = kernels.gemv_fp8(x.float(), ids, plan, gate_up, gate_up, gate_up_rows).chunk(2, dim=-1)
        out = kernels.gemv_fp8(nn.functional.silu(gate) * up, ids, plan, down, down, down_rows)
        return kernels.weighted_sum(out, weights.float())
    # no expert is left to the CPU (took_over 0); the rotation always runs (1)
    took_over, always = torch.arange(2, dtype=torch.int32, device=x.device).split(1)
    hidden = quantized_gemv(kind, kernels.rotate(x, always), ids, plan, took_over, False, gate_up, gate_up, gate_up_rows, codebooks[0])
    out = quantized_gemv(kind, kernels.rotate(hidden, always, swiglu=True), ids, plan, took_over, False, down, down, down_rows,
                         codebooks[1])
    return kernels.weighted_sum(out, weights.float())


def grouped_experts(sorted_x, groups, slot_banks, host_banks, rows, kind, codebooks, always):
    """[rows, H] float32: sorted_x's rows through their experts, one grouped kernel per bank (see kernels.grouped_int8);
    `always` is an int32 1 on the GPU."""
    if kind == "fp8":
        gate, up = kernels.grouped_fp8(sorted_x.float(), groups, slot_banks[0], host_banks[0], rows[0]).chunk(2, dim=-1)
        return kernels.grouped_fp8(nn.functional.silu(gate) * up, groups, slot_banks[1], host_banks[1], rows[1])
    hidden = quantized_grouped(kind, kernels.rotate(sorted_x.float(), always), groups, slot_banks[0], host_banks[0], rows[0],
                               codebooks[0])
    return quantized_grouped(kind, kernels.rotate(hidden, always, swiglu=True), groups, slot_banks[1], host_banks[1], rows[1],
                             codebooks[1])


def resident_prefill(x, topk_idx, topk_weight, banks, rows, kind, codebooks, num_experts):
    """Many rows x [T, H] through banks held whole on the GPU (as resident_decode): one grouped kernel per bank."""
    num_tokens, top_k = topk_idx.shape
    order = torch.argsort(topk_idx.reshape(-1))
    counts = torch.bincount(topk_idx.reshape(-1), minlength=num_experts).tolist()
    offsets = [0, *itertools.accumulate(counts)]
    groups = torch.tensor([(expert, expert, 0, offsets[expert], n) for expert, n in enumerate(counts) if n], dtype=torch.int32)
    always = torch.ones((1,), dtype=torch.int32, device=x.device)
    sorted_out = grouped_experts(x[order // top_k], groups.to(x.device), banks, banks, rows, kind, codebooks, always)
    out = torch.empty_like(sorted_out)
    out[order] = sorted_out * topk_weight.reshape(-1)[order].unsqueeze(1).float()
    return out.view(num_tokens, top_k, -1).sum(dim=1).to(x.dtype)


def gather_experts(weight, ids, num_experts):
    """Experts `ids` [k] (a device tensor) of a flat [E * out, in] bank as one [k * out, in] weight, on the GPU."""
    if not isinstance(weight, QuantizedTensor):
        return weight.view(num_experts, -1, weight.shape[-1]).index_select(0, ids).flatten(0, 1)
    rows = weight._qdata.shape[0] // num_experts
    picked = {n: v.view(num_experts, rows, *v.shape[1:]).index_select(0, ids).flatten(0, 1) for n, v in per_row_fields(weight).items()}
    qdata = picked.pop("_qdata")
    return QuantizedTensor(qdata, weight._layout_cls,
                           dataclasses.replace(weight._params, orig_shape=(len(ids) * rows, weight._params.orig_shape[-1]), **picked))


def dequantize_fp8(weight, scale, dtype):
    """Float8 rows [R, C] with one float32 scale per 128 x 128 block [R / 128, C / 128] -> dtype, as Fish's code does."""
    rows, cols = weight.shape
    blocks = weight.float().view(rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK)
    return (blocks * scale[:, None, :, None]).view(rows, cols).to(dtype)


def hadamard(size):
    """ConvRot's rotation: the regular Hadamard matrix of `size` (a power of 4), normalized; symmetric."""
    h4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]])
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / size ** 0.5


def experts_on_cpu(banks, experts, counts, x, rotation):
    """Rows x [rows, H] (bfloat16, already rotated; counts[i] rows for experts[i], in turn) through their int8 ConvRot
    experts on the CPU: [rows, H] float32. Only the matmuls go expert by expert."""
    gate_up, down = banks
    if len(experts) == 1:
        hidden = comfy.ops._swiglu_eager(gate_up.cpu_linear(x, experts[0]).float())
        return down.cpu_linear((hidden.view(len(x), -1, 256) @ rotation).flatten(1).to(torch.bfloat16), experts[0]).float()
    bounds = list(zip(experts, itertools.accumulate([0, *counts[:-1]]), itertools.accumulate(counts)))
    hidden = comfy.ops._swiglu_eager(torch.cat([gate_up.cpu_linear(x[a:b], expert) for expert, a, b in bounds]).float())
    hidden = (hidden.view(len(x), -1, 256) @ rotation).flatten(1).to(torch.bfloat16)
    return torch.cat([down.cpu_linear(hidden[a:b], expert) for expert, a, b in bounds]).float()


def experts_decode(x, gate_up, down, weights):
    """One row x [1, H] through k experts at once, as two matmuls. gate_up [k * 2I, H]: every expert sees the same
    row. down [k * H, I]: each expert's input row meets every expert's weights and only the diagonal blocks are
    kept; the weights are read once either way, which is what a one-row matmul costs."""
    k = weights.shape[0]
    hidden = comfy.ops._swiglu_eager(nn.functional.linear(x, gate_up).view(k, -1))
    out = nn.functional.linear(hidden, down).view(k, k, -1).diagonal(dim1=0, dim2=1).T
    return (out * weights[:, None].to(out.dtype)).sum(0, keepdim=True)


class HostBank:
    """One expert bank in pinned host memory: tensors whose first dimension holds each expert's rows in turn.
    `kernel` names the decode kernel that reads it ("int8", "fp8"), or None."""

    kernel = None

    def __init__(self, num_experts, fields):
        self.num_experts = num_experts
        self.fields = {n: pinned(t) for n, t in fields.items()}

    def rows(self, name):
        return self.fields[name].shape[0] // self.num_experts

    def take(self, fields, first, last):
        """Experts [first, last) of `fields`, host or device, laid out like this bank's."""
        return {n: t[first * self.rows(n):last * self.rows(n)] for n, t in fields.items()}

    def to_device(self, first, last, device):
        return {n: t.to(device, non_blocking=True) for n, t in self.take(self.fields, first, last).items()}


class QuantHostBank(HostBank):
    """A flat `comfy.ops.MoEExperts` bank (int8 ConvRot, W4A8, W6A8); its experts run as QuantizedTensors."""

    def __init__(self, module):
        weight = module.weight
        if not isinstance(weight, QuantizedTensor) or weight._qdata.ndim != 2:
            raise ValueError("Drama 3 keeps only flat quantized or FP8 expert banks in host memory")
        fields = per_row_fields(weight)
        super().__init__(module.num_experts, fields)
        self.shared = {n: v for n, v in tensor_fields(weight).items() if n not in fields}
        self.shared_on = {}
        self.layout, self.params, self.in_features = weight._layout_cls, weight._params, weight._params.orig_shape[-1]
        found = kernel_bank(weight)
        if found is not None:
            self.kernel = found[0]
            # the CPU's matmul takes its scales in the input's dtype
            self.scale_bf16 = self.fields["scale"].view(-1).to(torch.bfloat16)

    def codebook(self, device):
        """The bank's 4-bit codebook on `device` (W4A8), else None."""
        return self._shared(device).get("codebook")

    def cpu_linear(self, x, expert):
        """Rows x [M, K] (bfloat16, rotated) through int8 expert `expert` on the CPU, from host memory: [M, N]."""
        rows = self.rows("_qdata")
        return torch._weight_int8pack_mm(x, self.fields["_qdata"][expert * rows:(expert + 1) * rows],
                                         self.scale_bf16[expert * rows:(expert + 1) * rows])

    def _shared(self, device):
        # a CPU codebook would send every W4A8 matmul down comfy_kitchen's eager path
        if device not in self.shared_on:
            self.shared_on[device] = {n: v.to(device) for n, v in self.shared.items()}
        return self.shared_on[device]

    def weight(self, fields, device):
        fields = dict(fields)
        qdata = fields.pop("_qdata")
        return QuantizedTensor(qdata, self.layout, dataclasses.replace(self.params, orig_shape=(qdata.shape[0], self.in_features),
                                                                       **fields, **self._shared(device)))


class FP8HostBank(HostBank):
    """Experts as Fish ships them, float8 rows and block scales; on the host-lookup path their experts run in bf16."""

    kernel = "fp8"

    def __init__(self, weight, scale):
        super().__init__(weight.shape[0], {"weight": weight.view(torch.uint8).flatten(0, 1), "scale": scale.flatten(0, 1)})

    def weight(self, fields, device):
        return dequantize_fp8(fields["weight"].view(torch.float8_e4m3fn), fields["scale"], torch.bfloat16)


class CpuSide:
    """Where a decode step's layers hand experts to the CPU. A layer's graph copies its routing, its rotated input and
    its step here and records posted[layer], then waits (`kernels.await_cpu`) for flag[layer] to reach that step; the
    host (`ExpertStore.after_layer`) runs the experts and writes result[layer], then the flag. One set per layer: the
    host serves a layer while the next one is already launched."""

    def __init__(self, layers, k, hidden):
        self.plan = torch.empty((layers, k, 2), dtype=torch.int32, pin_memory=True)
        self.ids = torch.empty((layers, k), dtype=torch.long, pin_memory=True)
        self.weights = torch.empty((layers, k), dtype=torch.float32, pin_memory=True)
        self.rows = torch.empty((layers, 1, hidden), dtype=torch.bfloat16, pin_memory=True)
        self.step = torch.zeros((layers,), dtype=torch.int32, pin_memory=True)
        self.flag = torch.zeros((layers,), dtype=torch.int32, pin_memory=True)
        self.result = torch.empty((layers, hidden), dtype=torch.bfloat16, pin_memory=True)
        self.posted = [torch.cuda.Event(external=True) for _ in range(layers)]

    def post(self, layer, plan, ids, weights, rows, step):
        for target, source in ((self.plan[layer], plan), (self.ids[layer], ids), (self.weights[layer], weights),
                               (self.rows[layer], rows), (self.step[layer:layer + 1], step)):
            target.copy_(source, non_blocking=True)
        self.posted[layer].record()

    def wait(self, layer, plan, step, took_over):
        return kernels.await_cpu(plan, self.flag[layer:layer + 1], step, self.result[layer], AWAIT_US, took_over)


class ExpertStore:
    def __init__(self, cache_bytes=0):
        """`cache_bytes` 0 sizes the expert cache to the VRAM free when a generation starts."""
        self.banks = []
        self.cache = None
        self.cache_bytes = cache_bytes
        # set per generation: decode runs the kernels, never waiting on the host
        self.sync_free = False
        self.cpu = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="drama3-experts")
        # measured / modelled time of the CPU's and the GPU's share of a prompt's experts
        self.cpu_scale = self.gpu_scale = 1.0
        self.rotation = hadamard(256)
        # set per generation with int8 experts and a small cache: the CPU runs part of the missed experts, in a prompt's
        # pass and, on one GPU, in decode steps through `cpu_side`
        self.cpu_experts = False
        self.cpu_side = None
        # what a decode step's expert costs the CPU, as measured
        self.cpu_expert_us = CPU_EXPERT_US + CPU_ROW_US

    def add(self, experts):
        """Take a layer's experts into host memory; returns the module that replaces them in the model."""
        if isinstance(experts, FP8Experts):
            banks = (FP8HostBank(experts.gate_up_weight, experts.gate_up_scale), FP8HostBank(experts.down_weight, experts.down_scale))
        else:
            banks = (QuantHostBank(experts.gate_up_proj), QuantHostBank(experts.down_proj))
        self.banks.append(banks)
        return OffloadedExperts(self, len(self.banks) - 1)

    @contextlib.contextmanager
    def session(self, device, patchers, cpu_decode, rows, cpu_experts=True):
        """One generation's expert cache, sized to the VRAM free at its start less what the patchers' weights on
        this device still need (a dynamic model's weights arrive on first use, and must not find the cache there) and
        what its longest take needs for `rows` rows of prompt and frames. A session opened inside another shares its
        cache, so the takes of one script keep what earlier ones loaded.
        `cpu_decode`: every decode step calls `after_layer` after each layer (the model runs on this device alone).
        `cpu_experts`: the CPU may run part of the missed int8 experts."""
        if self.cache is not None:
            yield
            return
        loaded = [p.loaded_size() for p in patchers]
        pending = sum(p.model_size() - size for p, size in zip(patchers, loaded) if p.load_device == device)
        reserve = RESERVE_BASE + int(RESERVE_PER_ROW * rows)
        # other models (the codec, speech recognition) leave the VRAM to the cache; they load again when next used
        wanted = self.cache_bytes or sum(t.nbytes for banks in self.banks for bank in banks for t in bank.fields.values())
        comfy.model_management.free_memory(wanted + pending + reserve, device,
                                           [m for m in comfy.model_management.current_loaded_models if m.model in patchers])
        budget = max(0, comfy.model_management.get_free_memory(device) - pending - reserve)
        if self.cache_bytes:
            budget = min(budget, self.cache_bytes)
        self.cache = SlotPool(self.banks, budget, device)
        if sum(p.loaded_size() for p in patchers) < sum(loaded):
            # ComfyUI's dynamic VRAM counted less free VRAM than there is (its last reading was taken before the codec
            # was unloaded and torch's pool trimmed), made room for the cache by evicting weights of this model, and
            # keeps them out until the model is loaded again. Without this a whole generation streams them from RAM,
            # at 250 ms a frame
            time.sleep(EVICTION_PAUSE)
            comfy.model_management.load_models_gpu(patchers)
        self.sync_free = (self.banks[0][0].kernel is not None and not comfy.model_management.DISABLE_PINNED_MEMORY
                          and kernels.available(device))
        if self.sync_free and self.banks[0][0].kernel == "w4":
            # before any decode step is recorded as a graph, which cannot copy them in
            for bank in itertools.chain.from_iterable(self.banks):
                bank.codebook(device)
        # The CPU runs most missed int8 experts, beside the GPU, while the cache holds under half of the experts: per
        # decode step on the 3090, 29 vs 42 ms with an 8 GB cache, 35 vs 58 with 4 GB; on the 4090 34 vs 53 with 4 GB. With
        # a larger cache the GPU reading its misses over PCIe is as fast, since those reads also fill the cache (4090,
        # 17 GB: 17.8 against 19.4 with the CPU side), and a seed's take repeats exactly: the CPU's arithmetic differs
        # from the kernels' in the last bits
        self.cpu_experts = (cpu_experts and self.sync_free and self.banks[0][0].kernel == "int8"
                            and 2 * self.cache.count < len(self.banks) * self.banks[0][0].num_experts)
        if cpu_decode and self.cpu_experts:
            self.cpu_side = CpuSide(len(self.banks), MAX_EXPERTS_PER_ROW, self.banks[0][0].in_features)
        try:
            yield
        finally:
            self.cache.report()
            self.cache = None
            self.sync_free = False
            self.cpu_experts = False
            self.cpu_side = None
            comfy.model_management.soft_empty_cache()

    def between_steps(self):
        """Between decode steps with the CPU side: copy the experts it ran into the slots they reserved, and give the GPU
        the share of missed experts that lets both sides finish together."""
        self.cache.promote()
        pcie = self.cache.expert_bytes / PCIE_BYTES_PER_US
        self.cache.share.fill_(self.cpu_expert_us / (self.cpu_expert_us + pcie))

    def forward(self, layer, x, topk_idx, topk_weight):
        banks = self.banks[layer]
        if x.shape[0] == 1 and self.sync_free and x.device == self.cache.device:
            return self.decode(layer, banks, x, topk_idx[0], topk_weight[0])
        if x.shape[0] == 1 and self.cache is not None:
            slots = self.cache.slots(layer, topk_idx[0].tolist(), banks)
            gate_up, down = (bank.weight(self.cache.gather(index, slots), x.device) for index, bank in enumerate(banks))
            return experts_decode(x, gate_up, down, topk_weight[0])
        return self.prefill(layer, banks, x, topk_idx, topk_weight)

    def decode(self, layer, banks, x, ids, weights):
        """One row x [1, H] through its experts `ids` [k] with the kernels: gate_up, SwiGLU, down, weighted sum."""
        gate_up, down = banks
        cache, side = self.cache, self.cpu_side
        plan = kernels.resolve(ids, layer, len(self.banks), gate_up.num_experts, HALF_LIFE, side is not None, ADMIT,
                               *cache.device_state())
        slots_gate_up, slots_down = cache.fields
        weights = weights.float()
        if gate_up.kernel == "fp8":
            gate, up = kernels.gemv_fp8(x.float(), ids, plan, slots_gate_up, gate_up.fields, gate_up.rows("weight")).chunk(2, dim=-1)
            out = kernels.gemv_fp8(nn.functional.silu(gate) * up, ids, plan, slots_down, down.fields, down.rows("weight"))
            return kernels.weighted_sum(out, weights)
        kind = gate_up.kernel

        def experts(late):
            """[k, H]: each expert's output, unweighted."""
            hidden = quantized_gemv(kind, rotated, ids, plan, cache.took_over, late, slots_gate_up, gate_up.fields,
                                    gate_up.rows("_qdata"), gate_up.codebook(x.device))
            hidden = kernels.rotate(hidden, cache.took_over if late else cache.always, swiglu=True)
            return quantized_gemv(kind, hidden, ids, plan, cache.took_over, late, slots_down, down.fields,
                                  down.rows("_qdata"), down.codebook(x.device))

        rotated = kernels.rotate(x, cache.always)
        if side is None:
            return kernels.weighted_sum(experts(False), weights)
        # the CPU reads the same rotated row; its experts run there while the GPU runs the others
        side.post(layer, plan, ids, weights, rotated.to(torch.bfloat16), cache.step)
        out = weights @ experts(False) + side.wait(layer, plan, cache.step, cache.took_over)
        # a late host: the GPU runs the CPU's experts itself
        return (out + weights @ experts(True)).to(x.dtype)

    def after_layer(self, layer):
        """The host's part of decode layer `layer`, already launched: wait for its routing, run the experts it left to
        the CPU, and hand the result to the layer, which waits for it on the GPU."""
        side = self.cpu_side
        side.posted[layer].synchronize()
        plan = side.plan[layer].tolist()
        on_cpu = [e for e, (_, code) in enumerate(plan) if code == 2]
        if not on_cpu:
            return
        ids = side.ids[layer].tolist()
        for e in on_cpu:
            if plan[e][0] >= 0:
                self.cache.reserved.append((layer, ids[e], plan[e][0]))
        started = time.perf_counter()
        out = experts_on_cpu(self.banks[layer], [ids[e] for e in on_cpu], [1] * len(on_cpu), side.rows[layer].expand(len(on_cpu), -1),
                             self.rotation)
        side.result[layer].copy_(side.weights[layer, on_cpu] @ out)
        side.flag[layer] = int(side.step[layer])
        self.cpu_expert_us = 0.9 * self.cpu_expert_us + 0.1 * 1e6 * (time.perf_counter() - started) / len(on_cpu)

    def prefill(self, layer, banks, x, topk_idx, topk_weight):
        """Many rows: each expert the rows use runs once on its rows."""
        num_tokens, top_k = topk_idx.shape
        order = torch.argsort(topk_idx.reshape(-1))
        counts = torch.bincount(topk_idx.reshape(-1), minlength=banks[0].num_experts).tolist()
        sorted_x = x[order // top_k]
        if self.sync_free:
            sorted_out = self.prefill_kernels(layer, banks, sorted_x, counts)
        else:
            sorted_out = self.prefill_host(layer, banks, sorted_x, counts)
        out = torch.empty_like(sorted_out)
        out[order] = sorted_out * topk_weight.reshape(-1)[order].unsqueeze(1)
        return out.view(num_tokens, top_k, -1).sum(dim=1)

    def prefill_kernels(self, layer, banks, sorted_x, counts):
        """The rows `sorted_x` (sorted by expert, counts[e] rows each) through their experts, unweighted. The cached
        experts run from their slots and the most used of the others fill free slots on the way, as one grouped kernel
        per bank that reads the rest from host memory; int8 experts with few rows run on the CPU meanwhile."""
        gate_up, down = banks
        cache = self.cache
        device = sorted_x.device
        offsets = [0, *itertools.accumulate(counts)]
        used = [expert for expert, n in enumerate(counts) if n]
        slots = cache.use(layer, used)
        rest = sorted((expert for expert in used if expert not in slots), key=counts.__getitem__, reverse=True)
        free = cache.free_slots()
        fills = rest[:min(cache.count // len(self.banks), len(free))]
        slots.update((expert, cache.assign(slot, layer, expert)) for expert, slot in zip(fills, free))
        fills = set(fills)
        on_cpu = []
        if self.cpu_experts:
            on_cpu, cpu_us, gpu_us = self.split(rest[len(fills):], counts, cache.expert_bytes)
        cpu_set = set(on_cpu)
        groups = [(expert, slots.get(expert, -1), int(expert in fills), offsets[expert], counts[expert]) for expert in used if expert not in cpu_set]
        job = None
        if on_cpu:
            index = torch.cat([torch.arange(offsets[expert], offsets[expert + 1]) for expert in on_cpu]).to(device)
            rows = (sorted_x[index].float().view(len(index), -1, 256) @ hadamard(256).to(device)).flatten(1)
            job = self.cpu.submit(timed, experts_on_cpu, banks, on_cpu, [counts[expert] for expert in on_cpu],
                                  rows.to(torch.bfloat16).cpu(), self.rotation)
        out = torch.empty((len(sorted_x), sorted_x.shape[1]), dtype=torch.float32, device=device)
        if groups:
            groups = torch.tensor(groups, dtype=torch.int32).to(device)
            started, ended = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started.record()
            bank_rows = [bank.rows("weight" if gate_up.kernel == "fp8" else "_qdata") for bank in banks]
            codebooks = [bank.codebook(device) if gate_up.kernel == "w4" else None for bank in banks]
            out = grouped_experts(sorted_x, groups, cache.fields, [bank.fields for bank in banks], bank_rows, gate_up.kernel,
                                  codebooks, cache.always)
            ended.record()
        if job is not None:
            result, cpu_seconds = job.result()
            out[index] = result.to(device, torch.float32)
            # the next pass's split follows what this one measured
            self.cpu_scale = 0.5 * self.cpu_scale + 0.5 * 1e6 * cpu_seconds / cpu_us
            if groups is not None and len(groups) and gpu_us:
                ended.synchronize()
                self.gpu_scale = 0.5 * self.gpu_scale + 0.5 * 1e3 * started.elapsed_time(ended) / gpu_us
        return out.to(sorted_x.dtype)

    def split(self, experts, counts, expert_bytes):
        """(experts for the CPU, its modelled µs, the GPU's modelled µs) of `experts`, most rows first: each goes to
        whichever side would finish it sooner, the GPU reading it from host memory over PCIe."""
        transfer = self.gpu_scale * expert_bytes / PCIE_BYTES_PER_US
        on_cpu, cpu_us, gpu_us = [], 0.0, 0.0
        for expert in experts:
            cost = self.cpu_scale * (CPU_EXPERT_US + CPU_ROW_US * counts[expert])
            if cpu_us + cost <= gpu_us + transfer:
                on_cpu.append(expert)
                cpu_us += cost
            else:
                gpu_us += transfer
        return on_cpu, cpu_us / self.cpu_scale, gpu_us / self.gpu_scale

    def prefill_host(self, layer, banks, sorted_x, counts):
        """prefill_kernels without the kernels: a loop over the experts, streaming the uncached ones in and keeping
        the ones these rows use most, up to the layer's share of the cache."""
        gate_up, down = banks
        cache = self.cache
        offsets = [0, *itertools.accumulate(counts)]
        sorted_out = torch.empty_like(sorted_x)
        used = [expert for expert, n in enumerate(counts) if n]
        cached = cache.use(layer, used)
        # never into the slots this layer is about to read
        share = min(cache.count // len(self.banks), cache.room(layer))
        keep = set([expert for expert in sorted(used, key=counts.__getitem__, reverse=True) if expert not in cached][:share])
        for expert in used:
            if expert in cached:
                sources = [cache.slot_fields(index, cached[expert]) for index in range(len(banks))]
            elif expert in keep:
                slot = cache.insert(layer, expert, [bank.take(bank.fields, expert, expert + 1) for bank in banks])
                sources = [cache.slot_fields(index, slot) for index in range(len(banks))]
            else:
                sources = [bank.to_device(expert, expert + 1, sorted_x.device) for bank in banks]
            rows = slice(offsets[expert], offsets[expert + 1])
            hidden = comfy.ops._swiglu_eager(nn.functional.linear(sorted_x[rows], gate_up.weight(sources[0], sorted_x.device)))
            sorted_out[rows] = nn.functional.linear(hidden, down.weight(sources[1], sorted_x.device))
        return sorted_out


def timed(function, *args):
    """(function(*args), seconds it took)."""
    started = time.perf_counter()
    return function(*args), time.perf_counter() - started


class SlotPool:
    """A fixed number of expert slots on the GPU, filled from host memory, the least used expert out first.

    `table` maps layer * experts + expert (a key) to its slot, -1 (host memory only) or -2 - slot (the slot is being
    copied in, `pending`); `owner` maps a slot to its key (-1: free). `score` counts each key's uses, halving every
    HALF_LIFE passes, as of step `at` (pass * layers + layer; -1: never); `clock` ticks once per pass (a prompt or a
    decode step) and `reads` counts decode hits, experts filled into a slot and experts run on the CPU. They live on the
    host for a prompt's pass and the host-lookup decode, and on the GPU while the decode kernels run (`device_state`):
    `host` brings them back. With the CPU side, the slots decode steps reserve are copied in between them (`promote`)."""

    def __init__(self, banks, budget, device):
        self.banks = banks
        self.rows = [{n: bank.rows(n) for n in bank.fields} for bank in banks[0]]
        self.expert_bytes = sum(t[:bank.rows(n)].nbytes for bank in banks[0] for n, t in bank.fields.items())
        self.layers, self.num_experts = len(banks), banks[0][0].num_experts
        self.count = max(8, min(self.layers * self.num_experts, budget // self.expert_bytes))
        self.fields = [{n: torch.empty((self.count * bank.rows(n), *t.shape[1:]), dtype=t.dtype, device=device) for n, t in bank.fields.items()}
                       for bank in banks[0]]
        self.device = device
        self.table = torch.full((self.layers * self.num_experts,), -1, dtype=torch.int32)
        self.owner = torch.full((self.count,), -1, dtype=torch.int32)
        self.score = torch.zeros((self.layers * self.num_experts,), dtype=torch.float32)
        self.at = torch.full((self.layers * self.num_experts,), -1, dtype=torch.int32)
        self.pending = torch.zeros((self.count,), dtype=torch.int32)
        self.clock = torch.zeros((1,), dtype=torch.int32)
        self.reads = torch.zeros((3,), dtype=torch.int32)
        self.on_device = False
        self.resident = None
        # (layer, expert, slot) reserved since the last promote; (event, [(key, slot)]) copies in flight
        self.reserved, self.copying = [], []
        self.reserve = RESERVE_START
        self.budget = None
        self.copy_stream = None
        # the decode kernels' scratch on the GPU: a layer's step, whether the GPU took over the CPU's experts (see
        # kernels.await_cpu), a set flag, and the share of a layer's missed experts the GPU runs beside the CPU
        self.step, self.took_over, self.always = torch.tensor([[0], [0], [1]], dtype=torch.int32, device=device)
        self.share = torch.zeros((1,), dtype=torch.float32, device=device)

    def device_state(self):
        """(table, owner, score, at, pending, clock, budget, reads, step, share) on the GPU, where the decode kernels keep
        them."""
        if not self.on_device:
            budget = torch.tensor(self._budget(), dtype=torch.int32)
            if self.resident is None:
                self.resident, self.budget = [t.to(self.device) for t in self._state()], budget.to(self.device)
            else:
                # at the addresses recorded decode steps read: a prompt's pass between them (a line taking over a
                # row) brings the state to the host and back
                for there, here in zip(self.resident, self._state()):
                    there.copy_(here)
                self.budget.copy_(budget)
            self.table, self.owner, self.score, self.at, self.pending, self.clock, self.reads = self.resident
            self.on_device = True
        return self.table, self.owner, self.score, self.at, self.pending, self.clock, self.budget, self.reads, self.step, self.share

    def host(self):
        if self.on_device:
            self.settle()
            self.table, self.owner, self.score, self.at, self.pending, self.clock, self.reads = (t.cpu() for t in self._state())
            self.on_device = False

    def _state(self):
        return self.table, self.owner, self.score, self.at, self.pending, self.clock, self.reads

    def promote(self):
        """Between decode steps: map the slots whose copies have landed, then copy the experts reserved since the last
        call into theirs on a copy stream, which overlaps the next steps. The next step may reserve as many slots as
        the copies kept up with."""
        if not self.on_device:
            return
        landed = []
        while self.copying and self.copying[0][0].query():
            landed += self.copying.pop(0)[1]
        self._map(landed)
        lagging = bool(self.copying)
        if self.reserved:
            if self.copy_stream is None:
                self.copy_stream = torch.cuda.Stream(self.device)
            self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.copy_stream):
                for layer, expert, slot in self.reserved:
                    self._fill(slot, [bank.take(bank.fields, expert, expert + 1) for bank in self.banks[layer]])
                event = torch.cuda.Event()
                event.record()
            self.copying.append((event, [(layer * self.num_experts + expert, slot) for layer, expert, slot in self.reserved]))
            self.reserved = []
        self.reserve = max(RESERVE_MIN, self.reserve * 3 // 4) if lagging else min(RESERVE_MAX, self.reserve * 5 // 4 + 1)
        self.budget.copy_(torch.tensor(self._budget(), dtype=torch.int32), non_blocking=True)

    def _budget(self):
        """[slots a decode step may reserve, at most per layer]: spread over the layers, with room for busy ones."""
        return [self.reserve, max(1, -(-2 * self.reserve // self.layers))]

    def settle(self):
        """Every reserved slot copied in and mapped, before the host takes the table."""
        self.promote()
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        self._map([pair for _, pairs in self.copying for pair in pairs])
        self.copying = []

    def _map(self, pairs):
        if pairs:
            keys, slots = zip(*pairs)
            keys = torch.tensor(keys, dtype=torch.long).pin_memory().to(self.device, non_blocking=True)
            slots = torch.tensor(slots, dtype=torch.int32).pin_memory().to(self.device, non_blocking=True)
            self.table[keys] = slots
            self.pending[slots.long()] = 0

    def _now(self, layer):
        """`layer`'s step in this pass; the first layer starts a new pass."""
        if layer == 0:
            self.clock += 1
        return int(self.clock) * self.layers + layer

    def _uses(self, keys, now):
        """Use counts of `keys` at step `now`."""
        at = self.at[keys]
        return torch.where(at < 0, 0.0, self.score[keys] * torch.exp2((at - now).float() / (HALF_LIFE * self.layers)))

    def _fill(self, slot, sources):
        for fields, rows, source in zip(self.fields, self.rows, sources):
            for name, target in fields.items():
                target[slot * rows[name]:(slot + 1) * rows[name]].copy_(source[name], non_blocking=True)

    def slot_fields(self, index, slot):
        """Bank `index`'s fields of the expert in `slot`, laid out like a host bank's expert."""
        return {n: t[slot * self.rows[index][n]:(slot + 1) * self.rows[index][n]] for n, t in self.fields[index].items()}

    def use(self, layer, experts):
        """{expert: slot} of the cached ones among `experts`; each of them counts a use in this pass."""
        self.host()
        now = self._now(layer)
        keys = torch.tensor(experts, dtype=torch.long) + layer * self.num_experts
        self.score[keys] = self._uses(keys, now) + 1
        self.at[keys] = now
        table = self.table[layer * self.num_experts:(layer + 1) * self.num_experts].tolist()
        return {expert: table[expert] for expert in experts if table[expert] >= 0}

    def free_slots(self):
        return (self.owner < 0).nonzero().flatten().tolist()

    def assign(self, slot, layer, expert):
        """Give `slot` to an expert of `layer`; its old expert leaves the table. The caller fills it."""
        key = layer * self.num_experts + expert
        old = int(self.owner[slot])
        if old >= 0:
            self.table[old] = -1
        self.owner[slot], self.table[key] = key, slot
        return slot

    def claim(self, layer, expert):
        """The slot of the least used cached expert (a free one first) that `layer` has not used in this pass."""
        keys = self.owner.long()
        now = int(self.clock) * self.layers + layer
        uses = torch.where(keys < 0, -1.0, self._uses(keys.clamp(min=0), now))
        uses[(keys >= 0) & (self.at[keys.clamp(min=0)] == now)] = float("inf")
        return self.assign(int(torch.argmin(uses)), layer, expert)

    def room(self, layer):
        """How many slots `layer` may still claim in this pass: those not holding an expert it used."""
        keys = self.owner.long()
        return int(((keys < 0) | (self.at[keys.clamp(min=0)] != int(self.clock) * self.layers + layer)).sum())

    def insert(self, layer, expert, sources):
        """A slot for an expert `layer` uses in this pass, filled from `sources` (host memory)."""
        slot = self.claim(layer, expert)
        self._fill(slot, sources)
        return slot

    def slots(self, layer, experts, banks):
        """The host-lookup decode: slots of the step's `experts`, missed ones copied in."""
        cached = self.use(layer, experts)
        slots = []
        for expert in experts:
            slot = cached.get(expert)
            if slot is None:
                slot = self.insert(layer, expert, [bank.take(bank.fields, expert, expert + 1) for bank in banks])
            slots.append(slot)
        self.reads += torch.tensor([len(cached), len(experts) - len(cached), 0], dtype=torch.int32)
        return torch.tensor(slots, device=self.device)

    def gather(self, index, slots):
        return {n: t.view(self.count, self.rows[index][n], *t.shape[1:]).index_select(0, slots).flatten(0, 1)
                for n, t in self.fields[index].items()}

    def report(self):
        self.host()
        hits, fills, on_cpu = self.reads.tolist()
        if hits + fills + on_cpu:
            logging.info("Drama 3 expert cache: %d experts (%.1f GiB), %.0f%% of %d expert reads hit; of the misses %d were "
                         "read over PCIe, %d run on the CPU", self.count, sum(t.nbytes for f in self.fields for t in f.values()) / 1024 ** 3,
                         100 * hits / (hits + fills + on_cpu), hits + fills + on_cpu, fills, on_cpu)


class FP8Experts(nn.Module):
    """Routed experts as Fish ships them: float8 e4m3 weights with a float32 scale per 128 x 128 block. A use turns
    only the experts it needs into the compute dtype."""

    def __init__(self, num_experts, hidden, intermediate, device=None, dtype=None):
        super().__init__()
        self.num_experts = num_experts
        self.dtype = dtype
        for name, rows, cols in (("gate_up", 2 * intermediate, hidden), ("down", hidden, intermediate)):
            self.register_parameter(f"{name}_weight", nn.Parameter(torch.empty((num_experts, rows, cols), dtype=torch.float8_e4m3fn, device=device), requires_grad=False))
            self.register_parameter(f"{name}_scale", nn.Parameter(torch.empty((num_experts, rows // FP8_BLOCK, cols // FP8_BLOCK), dtype=torch.float32, device=device), requires_grad=False))

    def sync_free(self, device):
        """Whether a decode row on `device` runs without waiting on the host (the kernels compile there)."""
        return kernels.available(device)

    def experts(self, name, ids):
        weight = getattr(self, f"{name}_weight").view(torch.uint8).index_select(0, ids).view(torch.float8_e4m3fn)
        return dequantize_fp8(weight.flatten(0, 1), getattr(self, f"{name}_scale").index_select(0, ids).flatten(0, 1), self.dtype)

    def forward(self, x, topk_idx, topk_weight):
        if kernels.available(x.device):
            banks = [{"weight": getattr(self, f"{name}_weight").view(torch.uint8).flatten(0, 1),
                      "scale": getattr(self, f"{name}_scale").flatten(0, 1)} for name in ("gate_up", "down")]
            rows = [getattr(self, f"{name}_weight").shape[1] for name in ("gate_up", "down")]
            if x.shape[0] == 1:
                return resident_decode(x, topk_idx[0], topk_weight[0], banks, rows, "fp8", (None, None))
            return resident_prefill(x, topk_idx, topk_weight, banks, rows, "fp8", (None, None), self.num_experts)
        if x.shape[0] == 1:
            return experts_decode(x, self.experts("gate_up", topk_idx[0]), self.experts("down", topk_idx[0]), topk_weight[0])
        num_tokens, top_k = topk_idx.shape
        order = torch.argsort(topk_idx.reshape(-1))
        counts = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).tolist()
        sorted_x = x[order // top_k]
        weight = topk_weight.reshape(-1)[order].unsqueeze(1)
        sorted_out = torch.empty_like(sorted_x)
        start = 0
        for expert, n in enumerate(counts):
            if n == 0:
                continue
            ids = torch.tensor([expert], device=x.device)
            rows = slice(start, start + n)
            hidden = comfy.ops._swiglu_eager(nn.functional.linear(sorted_x[rows], self.experts("gate_up", ids)))
            sorted_out[rows] = nn.functional.linear(hidden, self.experts("down", ids)) * weight[rows]
            start += n
        out = torch.empty_like(sorted_out)
        out[order] = sorted_out
        return out.view(num_tokens, top_k, -1).sum(dim=1)


class HostEmbedding(nn.Module):
    """The token embedding table left in host memory, beside the experts: a prompt or a frame reads a few rows of its
    gigabyte, which the expert cache puts to better use."""

    def __init__(self, embedding):
        super().__init__()
        self.table = embedding.weight.detach()

    def forward(self, ids, out_dtype=None):
        return self.table[ids.cpu()].to(ids.device, out_dtype)


class OffloadedExperts(nn.Module):
    """Stands in for a layer's experts when they stay in host memory."""

    def __init__(self, store, layer):
        super().__init__()
        self.store = store
        self.layer = layer

    def sync_free(self, device):
        """Whether a decode row on `device` runs without waiting on the host, in the current generation."""
        return self.store.sync_free and device == self.store.cache.device

    def forward(self, x, topk_idx, topk_weight):
        return self.store.forward(self.layer, x, topk_idx, topk_weight)
