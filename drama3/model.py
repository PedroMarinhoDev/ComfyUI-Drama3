"""Fish Audio Drama 3: a Qwen3.5 MoE backbone that predicts each frame's main token, and a codebook tower that
predicts the frame's other codes.

A frame row is [main token id, code 0, ..., code N-1]. Text rows carry zeros for the codes; an audio row's main
token is the semantic token of its code 0.
"""
import math

import torch
import torch.nn as nn

import comfy_kitchen
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.text_encoders.llama
from comfy.ldm.flux.math import apply_rope1, rope
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import RMSNorm, moe_experts_forward, rope_matrix
from comfy.text_encoders.qwen35 import (GatedAttention, GatedDeltaNet, LinearKV, Qwen35Transformer, Qwen35TransformerBlock,
                                        torch_chunk_gated_delta_rule)
from . import kernels
from .offload import FP8Experts, experts_decode, gather_experts, kernel_bank, resident_decode, resident_prefill

# ComfyUI's Qwen3.5 decode changed after v0.41.2 (#17007): attention layers make their own KV caches and a DeltaNet
# decode step commits the previous step's token. v0.41 and older decode through FixedKVBias.
DEFERRED_DECODE = hasattr(LinearKV, "deferred_tracker")

if DEFERRED_DECODE:
    class RowKVCache(comfy.text_encoders.llama.FixedKVCache):
        """The KV cache of lines decoded in one batch. Their prompts differ in length, so each row writes at its own
        position (`position` [B]) and attends over its own length (`seqlen` [B]); the caches of a stack share both."""

        @staticmethod
        def shared(batch, device):
            return (torch.zeros((batch,), device=device, dtype=torch.int64), torch.zeros((batch,), device=device, dtype=torch.int32),
                    {"step": -1, "tree": None})

        def prepare(self, num_tokens):
            if self.tracker["step"] != self.index:
                self.tracker["step"] = self.index
                self.position.copy_(self.seqlen)
                self.seqlen.add_(num_tokens)

        def write(self, xk, xv):
            index = self.position.view(-1, 1, 1, 1).expand(-1, xk.shape[1], 1, xk.shape[3])
            self.key.scatter_(2, index, xk)
            self.value.scatter_(2, index, xv)


class SharedExpert(nn.Module):
    def __init__(self, hidden, intermediate, device=None, dtype=None, ops=None):
        super().__init__()
        self.gate_up_proj = ops.Linear(hidden, 2 * intermediate, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return comfy.ops.linear_input_act(self.down_proj, self.gate_up_proj(x), "swiglu")


class Experts(nn.Module):
    def __init__(self, num_experts, hidden, intermediate, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = ops.MoEExperts(num_experts, hidden, 2 * intermediate, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.MoEExperts(num_experts, intermediate, hidden, bias=False, device=device, dtype=dtype)

    def sync_free(self, device):
        """Whether a decode row on `device` runs without waiting on the host: the kernels read both banks."""
        return (kernels.available(device) and kernel_bank(self.gate_up_proj.weight) is not None
                and kernel_bank(self.down_proj.weight) is not None)

    def forward(self, x, topk_idx, topk_weight):
        with self.gate_up_proj.bank_resident(x) as gate_up, self.down_proj.bank_resident(x) as down:
            weights = gate_up._resident_bank[0], down._resident_bank[0]
            found = [kernel_bank(weight) for weight in weights]
            if None not in found and kernels.available(x.device):
                # the kernels read the experts in place: one row, or a prompt's rows grouped by expert
                kind, banks = found[0][0], [fields for _, fields in found]
                rows = [bank["_qdata"].shape[0] // self.num_experts for bank in banks]
                codebooks = [weight._params.codebook for weight in weights] if kind == "w4" else (None, None)
                if x.shape[0] == 1:
                    return resident_decode(x, topk_idx[0], topk_weight[0], banks, rows, kind, codebooks)
                return resident_prefill(x, topk_idx, topk_weight, banks, rows, kind, codebooks, self.num_experts)
            if x.shape[0] == 1:
                # other formats: gather its experts on the GPU and run them as one batch
                return experts_decode(x, *(gather_experts(weight, topk_idx[0], self.num_experts) for weight in weights), topk_weight[0])
        return moe_experts_forward(x, topk_idx, topk_weight, self.num_experts, self.gate_up_proj, self.down_proj,
                                   comfy.ops._swiglu_eager)


class SparseMoeBlock(nn.Module):
    """Softmax router over all experts, renormalized top-k, plus a sigmoid-gated shared expert."""

    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        hidden = config.text.hidden_size
        self.top_k = config.num_experts_per_tok
        self.gate = ops.Linear(hidden, config.num_experts, bias=False, device=device, dtype=dtype)
        if config.fp8_experts:
            self.experts = FP8Experts(config.num_experts, hidden, config.moe_intermediate_size, device=device, dtype=dtype)
        else:
            self.experts = Experts(config.num_experts, hidden, config.moe_intermediate_size, device=device, dtype=dtype, ops=ops)
        self.shared_expert = SharedExpert(hidden, config.shared_expert_intermediate_size, device=device, dtype=dtype, ops=ops)
        self.shared_expert_gate = ops.Linear(hidden, 1, bias=False, device=device, dtype=dtype)

    def forward(self, x, norm=None):
        # after v0.41.2 the block hands its pre-norm to the MLP
        if norm is not None:
            x = norm(x)
        if x.shape[0] > 1 and x.shape[1] == 1:
            # the decode rows of lines batched together: each through the one-row kernels
            return torch.cat([self(x[row:row + 1]) for row in range(x.shape[0])])
        flat = x.reshape(-1, x.shape[-1])
        logits = self.gate(flat)
        if flat.shape[0] == 1 and kernels.available(flat.device):
            # a decode row: one kernel in place of the softmax, the top-k and the renormalization
            topk_idx, topk_weight = kernels.route(logits, self.top_k)
        else:
            topk_weight, topk_idx = torch.softmax(logits.float(), dim=-1).topk(self.top_k, dim=-1)
            topk_weight = (topk_weight / topk_weight.sum(dim=-1, keepdim=True)).to(flat.dtype)
        out = self.experts(flat, topk_idx, topk_weight)
        out = out + torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return out.view_as(x)


class DeltaNet(GatedDeltaNet):
    """ComfyUI's GatedDeltaNet starts from an empty state on any call of more than a few rows. This one also continues
    from the state its cache holds, so a prompt's rows can run in two calls: the rows lines share once, kept, then
    each line's own (`decode_lines`). The second call needs more rows than a decode step takes."""

    def forward(self, x, past_key_value=None, **kwargs):
        # v0.41 and older never get here with rows to continue: only lines decoded together run a prompt in two calls
        if not DEFERRED_DECODE or past_key_value is None or past_key_value.index == 0 or past_key_value.decode_step(x.shape[1]):
            return super().forward(x, past_key_value=past_key_value, **kwargs)
        batch, rows, _ = x.shape
        qkvz = self.combined_qkvz.forward(x)
        if qkvz is NotImplemented:
            proj, z = self.in_proj_qkv(x), self.in_proj_z(x)
        else:
            proj, z = qkvz.split((self.key_dim * 2 + self.value_dim, self.value_dim), dim=-1)
        beta, a = self.in_proj_b(x).sigmoid(), self.in_proj_a(x)
        # the conv reads the columns before these rows from the cache
        state = past_key_value.conv_state
        mixed = torch.cat((state, proj.transpose(1, 2)), dim=-1)
        state.copy_(mixed[:, :, -state.shape[-1]:])
        mixed = nn.functional.silu(self.conv1d(mixed)[:, :, state.shape[-1]:state.shape[-1] + rows]).transpose(1, 2)
        query, key, value = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        g = past_key_value.g_decay * nn.functional.softplus(a.float() + past_key_value.dt_bias)
        query = query.reshape(batch, rows, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch, rows, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch, rows, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            repeats = self.num_value_heads // self.num_key_heads
            query, key = query.repeat_interleave(repeats, dim=2), key.repeat_interleave(repeats, dim=2)
        out, last = torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=past_key_value.recurrent_state,
                                                 output_final_state=True)
        past_key_value.recurrent_state.copy_(last.to(past_key_value.recurrent_state.dtype))
        out = self.norm(out.reshape(-1, self.value_head_dim), z.reshape(-1, self.value_head_dim))
        return self.out_proj(out.reshape(batch, rows, -1)), past_key_value


class Drama3Block(Qwen35TransformerBlock):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        nn.Module.__init__(self)
        text = config.text
        self.layer_type = text.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = DeltaNet(text, device=device, dtype=dtype, ops=ops)
        else:
            self.self_attn = GatedAttention(text, device=device, dtype=dtype, ops=ops)
        self.mlp = SparseMoeBlock(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps, add=True, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps, add=True, device=device, dtype=dtype)


class Drama3Backbone(Qwen35Transformer):
    def __init__(self, config, device=None, dtype=None, ops=None):
        nn.Module.__init__(self)
        text = config.text
        self.config = text
        self.embed_tokens = ops.Embedding(text.vocab_size, text.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(Drama3Block(config, i, device=device, dtype=dtype, ops=ops) for i in range(text.num_hidden_layers))
        self.norm = RMSNorm(text.hidden_size, eps=text.rms_norm_eps, add=True, device=device, dtype=dtype)

    def init_caches(self, capacity, layers, devices, dtype):
        """Per-layer decode state on each layer's device: DeltaNet conv and recurrent state, or a KV cache."""
        text = self.config
        conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim + text.linear_num_value_heads * text.linear_value_head_dim
        shared, trackers = {}, {}
        caches = []
        for layer, device in zip(layers, devices):
            if layer.layer_type == "linear_attention":
                recurrent = torch.zeros((1, text.linear_num_value_heads, text.linear_key_head_dim, text.linear_value_head_dim), device=device, dtype=torch.float32)
                conv = torch.zeros((1, conv_dim, text.conv_kernel_size - 1), device=device, dtype=dtype)
                cache = LinearKV(conv, recurrent, 0, None, None)
                if DEFERRED_DECODE:
                    if device not in trackers:
                        trackers[device] = LinearKV.deferred_tracker(device)
                    cache.qkv_buf, cache.proj_buf, cache.gates_buf, cache.sumsq_buf = comfy_kitchen.gated_delta_deferred_buffers(
                        1, conv_dim, text.linear_num_value_heads, text.linear_num_key_heads, dtype, device)
                    cache.ctl, cache.ctl_tracker = trackers[device]["ctl"], trackers[device]
                caches.append(cache)
            elif DEFERRED_DECODE:
                caches.append(layer.self_attn.init_kv_cache(1, capacity, device, dtype, shared.setdefault(device, {})))
            else:
                if device not in shared:
                    shared[device] = comfy.text_encoders.llama.FixedKVBias.shared(capacity, device, dtype)
                caches.append(comfy.text_encoders.llama.FixedKVBias.zeros(1, text.num_key_value_heads, capacity, text.head_dim, device, dtype, shared[device]))
        return caches

    def init_batch_caches(self, batch, capacity, layers, device, dtype):
        """(caches, side) for `batch` lines decoded together on one device: the decode state of all rows, and what
        loading a line into a row needs (`row_caches`): a tracker and one-row side buffers for the DeltaNet layers."""
        text = self.config
        conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim + text.linear_num_value_heads * text.linear_value_head_dim
        shared, tracker = RowKVCache.shared(batch, device), LinearKV.deferred_tracker(device)
        caches, buffers = [], []
        for layer in layers:
            if layer.layer_type == "linear_attention":
                cache = LinearKV(torch.zeros((batch, conv_dim, text.conv_kernel_size - 1), device=device, dtype=dtype),
                                 torch.zeros((batch, text.linear_num_value_heads, text.linear_key_head_dim, text.linear_value_head_dim),
                                             device=device, dtype=torch.float32), 0, None, None)
                cache.qkv_buf, cache.proj_buf, cache.gates_buf, cache.sumsq_buf = comfy_kitchen.gated_delta_deferred_buffers(
                    batch, conv_dim, text.linear_num_value_heads, text.linear_num_key_heads, dtype, device)
                cache.ctl, cache.ctl_tracker = tracker["ctl"], tracker
                buffers.append(comfy_kitchen.gated_delta_deferred_buffers(1, conv_dim, text.linear_num_value_heads, text.linear_num_key_heads,
                                                                         dtype, device))
            else:
                cache = RowKVCache.zeros(batch, text.num_key_value_heads, capacity, text.head_dim, device, dtype, shared)
            caches.append(cache)
        return caches, (LinearKV.deferred_tracker(device), buffers)

    @staticmethod
    def row_caches(caches, index, side):
        """One-row caches over row `index` of the batch's storage, for a line's prompt pass and the decode step of its
        last prompt row; `adopt_row` then hands the row to the batch."""
        tracker, buffers = side
        tracker["step"] = None
        kv, buffers = comfy.text_encoders.llama.FixedKVCache.shared(1, caches[0].key.device), iter(buffers)
        row = []
        for cache in caches:
            if isinstance(cache, LinearKV):
                row.append(LinearKV(cache.key[index:index + 1], cache.value[index:index + 1], 0, None, None,
                                    cache.g_decay, cache.dt_bias, cache.norm_weight))
                row[-1].qkv_buf, row[-1].proj_buf, row[-1].gates_buf, row[-1].sumsq_buf = next(buffers)
                row[-1].ctl, row[-1].ctl_tracker = tracker["ctl"], tracker
            else:
                row.append(comfy.text_encoders.llama.FixedKVCache(cache.key[index:index + 1], cache.value[index:index + 1], 0, *kv))
        return row

    @staticmethod
    def adopt_row(caches, row, index):
        """The batch takes row `index` as its one-row caches left it: its length and, in the DeltaNet layers, the token
        of its last step as the one the batch's next step commits (a decode step leaves its token to the next one, and
        the batch commits one token for every row). Returns the row's length."""
        for cache, one in zip(caches, row):
            cache.index = max(cache.index, 1)
            if isinstance(cache, LinearKV):
                if cache.g_decay is None:
                    # the layer's constants, which a prompt's pass makes anew: recorded steps read the first ones
                    cache.g_decay, cache.dt_bias, cache.norm_weight = one.g_decay, one.dt_bias, one.norm_weight
                side, own = cache.ctl_tracker["parity"], one.ctl_tracker["parity"]
                for target, source in ((cache.qkv_buf, one.qkv_buf), (cache.proj_buf, one.proj_buf), (cache.gates_buf, one.gates_buf),
                                       (cache.sumsq_buf, one.sumsq_buf)):
                    target[side, index].copy_(source[own, 0])
                cache.uncommitted = 1
            else:
                cache.seqlen[index] = one.index
        return row[0].index

    @staticmethod
    def empty_row(caches, index):
        """Row `index` holds no line: its attention caches start over, so the tokens it still writes stay in bounds."""
        next(cache for cache in caches if not isinstance(cache, LinearKV)).seqlen[index] = 0

    def run_layers(self, x, caches, layers, devices, decode_buffers=None, after_layer=None, ahead=False, positions=None):
        """Hidden states [1, T, H] after `layers` (the whole stack, in order) for the rows `x` [1, T, H] continue,
        each layer on its device; the result is on the last one. Not final-normed.

        A whole prompt runs in one call: the DeltaNet layers restart their state on any call of more than six rows.
        With `decode_buffers` (`init_decode_buffers`, one device) a decode row runs each layer as a CUDA graph, through
        the buffers, whose hidden state is the result. `after_layer(index)` runs on the host after each layer is
        launched, or with `ahead` after the next one is (the layer waits for it on the GPU). `positions` [B, 1]: the
        position of each row of a batch of decode rows (lines decoded together, one device, with `decode_buffers`).
        """
        past_len = caches[0].index
        start = 0
        while start < len(layers):
            device = devices[start]
            end = start
            while end < len(layers) and devices[end] == device:
                end += 1
            with comfy.model_management.cuda_device_context(device):
                x = self._segment(x.to(device), layers[start:end], caches[start:end], device, past_len, decode_buffers,
                                  after_layer and (lambda index, start=start: after_layer(start + index)), ahead, positions)
            start = end
        return x

    def _segment(self, x, layers, caches, device, past_len, decode_buffers, after_layer, ahead, positions):
        seq_len = x.shape[1]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, past_len + seq_len), torch.finfo(x.dtype).min / 4, device=device, dtype=x.dtype).triu_(past_len + 1)
        if positions is None:
            freqs = self.compute_freqs_cis(torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0), device)
        else:
            # one row at a time: compute_freqs_cis reads a leading dimension above one as image axes
            freqs = torch.cat([rope_matrix(self.compute_freqs_cis(row[None], device)) for row in positions])
        if decode_buffers is not None:
            # the layers write their output into x in place, so graph replays land in the buffer
            hidden_buffer, rotary_buffer = decode_buffers
            hidden_buffer.copy_(x)
            rotary_buffer.copy_(rope_matrix(freqs))
            x, freqs = hidden_buffer, rotary_buffer
        attention = optimized_attention_for_device(device, mask=mask is not None, small_input=True)
        queue = comfy.model_prefetch.make_prefetch_queue(list(layers), device, {"prefetch_dynamic_vbars": True})
        served = 0
        for index, (layer, cache) in enumerate(zip(layers, caches)):
            cache.prepare(seq_len)

            def core(layer=layer, cache=cache):
                nonlocal x
                x = layer(x, attention_mask=mask, freqs_cis=freqs, optimized_attention=attention, past_key_value=cache)[0]

            comfy.model_prefetch.prefetch_queue_pop(queue, device, layer, x.dtype, core=core, enable_graph=decode_buffers is not None,
                                                    malloc_scope="block")
            if after_layer is not None:
                while served < index + (not ahead):
                    after_layer(served)
                    served += 1
            cache.advance(seq_len)
        comfy.model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="block")
        while after_layer is not None and served < len(layers):
            after_layer(served)
            served += 1
        return x


class TowerAttention(nn.Module):
    """Gated GQA over every earlier row, q/k norm, interleaved partial RoPE. wqkv rows are [Q | gate | K | V]."""

    def __init__(self, cfg, device=None, dtype=None, ops=None):
        super().__init__()
        self.cfg = cfg
        q_width, kv_width = cfg.n_head * cfg.head_dim, cfg.n_local_heads * cfg.head_dim
        self.split = (q_width, q_width, kv_width, kv_width)
        self.wqkv = ops.Linear(cfg.dim, 2 * q_width + 2 * kv_width, bias=False, device=device, dtype=dtype)
        self.wo = ops.Linear(q_width, cfg.dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)

    def rotate(self, x, freqs):
        width = self.cfg.rotary_dim
        return torch.cat((apply_rope1(x[..., :width].contiguous(), freqs), x[..., width:]), dim=-1)

    def forward(self, x, context, cache):
        freqs, mask, attention = context
        cfg = self.cfg
        length = x.shape[1]
        q, gate, k, v = self.wqkv(x).split(self.split, dim=-1)
        q = self.rotate(self.q_norm(q.view(-1, length, cfg.n_head, cfg.head_dim)).transpose(1, 2), freqs)
        k = self.rotate(self.k_norm(k.view(-1, length, cfg.n_local_heads, cfg.head_dim)).transpose(1, 2), freqs)
        v = v.view(-1, length, cfg.n_local_heads, cfg.head_dim).transpose(1, 2)
        if mask is None and DEFERRED_DECODE:
            cache.write(k, v)
            out = cache.attend(q, k, v)
        elif mask is None:
            out = cache.decode(q, k, v, cfg.n_local_heads)
        else:
            k, v = cache.append(k, v)
            out = attention(q, k, v, cfg.n_head, mask=mask, skip_reshape=True, enable_gqa=True)
        return self.wo(out * torch.sigmoid(gate))


class TowerMLP(nn.Module):
    def __init__(self, cfg, device=None, dtype=None, ops=None):
        super().__init__()
        self.w13 = ops.Linear(cfg.dim, 2 * cfg.intermediate_size, bias=False, device=device, dtype=dtype)
        self.w2 = ops.Linear(cfg.intermediate_size, cfg.dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return comfy.ops.linear_input_act(self.w2, self.w13(x), "swiglu")


class TowerLayer(nn.Module):
    """Mix a code embedding into the hidden state (MTP style), then one pre-norm attention + MLP block."""

    def __init__(self, cfg, device=None, dtype=None, ops=None):
        super().__init__()
        self.embedding = ops.Embedding(cfg.vocab_size, cfg.dim, device=device, dtype=dtype)
        self.enorm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.hnorm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.eh_proj = ops.Linear(2 * cfg.dim, cfg.dim, bias=False, device=device, dtype=dtype)
        self.attention = TowerAttention(cfg, device=device, dtype=dtype, ops=ops)
        self.attention_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)
        self.feed_forward = TowerMLP(cfg, device=device, dtype=dtype, ops=ops)
        self.final_layernorm = RMSNorm(cfg.dim, eps=cfg.norm_eps, add=True, device=device, dtype=dtype)

    def forward(self, hidden, codes, semantic, context, cache, shared_input=None):
        embedded = self.embedding(codes.clamp(0, self.embedding.num_embeddings - 1), out_dtype=hidden.dtype)
        embedded = torch.where(semantic[..., None], embedded, 0.0)
        if shared_input is not None:
            embedded = embedded + shared_input
        mixed = self.eh_proj(torch.cat((self.enorm(embedded), self.hnorm(hidden)), dim=-1))
        x = mixed + self.attention(self.attention_norm(mixed), context, cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return self.final_layernorm(x)


class Drama3Tower(nn.Module):
    """Predicts codes 1..N-1 of a frame from the backbone's hidden state and code 0, one layer per code, each
    attending over all earlier rows through its own KV cache. Row p of a prompt is labelled with frame p + 1."""

    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        cfg = self.cfg = config.tower
        self.semantic_start = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.codebook_embeddings = ops.Embedding(cfg.vocab_size * cfg.num_codebooks, cfg.dim, device=device, dtype=dtype)
        self.shared_layers = nn.ModuleList(TowerLayer(cfg, device=device, dtype=dtype, ops=ops) for _ in range(cfg.shared_layers))
        self.residual_layers = nn.ModuleList(TowerLayer(cfg, device=device, dtype=dtype, ops=ops) for _ in range(cfg.num_codebooks - 1))
        self.heads = nn.ModuleList(ops.Linear(cfg.dim, cfg.vocab_size, bias=False, device=device, dtype=dtype) for _ in range(cfg.num_codebooks - 1))

    def is_semantic(self, tokens):
        return (tokens >= self.semantic_start) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames, first, end, dtype):
        """Summed embeddings of codes first..end-1 of each frame row."""
        offsets = torch.arange(first, end, device=frames.device) * self.cfg.vocab_size
        return self.codebook_embeddings(frames[:, 1 + first:1 + end] + offsets, out_dtype=dtype).sum(dim=-2)

    def embed_backbone(self, frames, text_embeddings):
        """Backbone input: an audio row adds the embeddings of codes 1..K to its main token's."""
        count = self.cfg.backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count, text_embeddings.dtype).to(text_embeddings.device)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[:, 0]).to(text_embeddings.device)
        return torch.where(semantic[:, None], mixed, text_embeddings)

    def init_caches(self, capacity, device, dtype):
        cfg = self.cfg
        if DEFERRED_DECODE:
            cache, shared = comfy.text_encoders.llama.FixedKVCache, comfy.text_encoders.llama.FixedKVCache.shared(1, device)
        else:
            cache, shared = comfy.text_encoders.llama.FixedKVBias, comfy.text_encoders.llama.FixedKVBias.shared(capacity, device, dtype)
        return [cache.zeros(1, cfg.n_local_heads, capacity, cfg.head_dim, device, dtype, shared)
                for _ in range(len(self.shared_layers) + len(self.residual_layers))]

    def init_batch_caches(self, batch, capacity, device, dtype):
        """The decode state of `batch` lines decoded together, as Drama3Backbone.init_batch_caches."""
        cfg = self.cfg
        shared = RowKVCache.shared(batch, device)
        return [RowKVCache.zeros(batch, cfg.n_local_heads, capacity, cfg.head_dim, device, dtype, shared)
                for _ in range(len(self.shared_layers) + len(self.residual_layers))]

    @staticmethod
    def row_caches(caches, index):
        kv = comfy.text_encoders.llama.FixedKVCache.shared(1, caches[0].key.device)
        return [comfy.text_encoders.llama.FixedKVCache(cache.key[index:index + 1], cache.value[index:index + 1], 0, *kv) for cache in caches]

    @staticmethod
    def adopt_row(caches, row, index):
        for cache in caches:
            cache.index = max(cache.index, 1)
        caches[0].seqlen[index] = row[0].index

    @staticmethod
    def empty_row(caches, index):
        caches[0].seqlen[index] = 0

    def context(self, positions, mask_rows, device, dtype):
        """(freqs, mask, attention) for rows at `positions` [T], or [B, 1] for a batch of decode rows; decode rows
        (mask_rows == 0) attend without a mask."""
        freqs = rope(positions if positions.dim() == 2 else positions.unsqueeze(0), self.cfg.rotary_dim, self.cfg.rope_base).unsqueeze(1)
        mask = None
        if mask_rows:
            past = int(positions[0])
            mask = torch.full((mask_rows, past + mask_rows), torch.finfo(dtype).min / 4, device=device, dtype=dtype).triu_(past + 1)
        return freqs, mask, optimized_attention_for_device(device, mask=mask is not None, small_input=True)

    def run(self, hidden, labels, frames, context, caches):
        """Run every layer on rows whose next frames are `labels` [T, 1 + N]; returns the last hidden state.

        `labels` holds code 0 of the next frame and, for the residual layers, the codes before the one each
        layer predicts. During decode the residual layers' codes are sampled one at a time instead (`step`)."""
        semantic = self.is_semantic(labels[:, 0])
        unseen = self.embedding_sum(frames, 1 + self.cfg.backbone_codebooks, self.cfg.num_codebooks, hidden.dtype)
        unseen = torch.where(self.is_semantic(frames[:, 0])[:, None], unseen, 0.0)
        for index, layer in enumerate(self.shared_layers):
            hidden = self._layer(layer, index, hidden, labels[:, 1], semantic, context, caches, unseen if index == 0 else None)
        for index, layer in enumerate(self.residual_layers):
            hidden = self._layer(layer, len(self.shared_layers) + index, hidden, labels[:, 1 + index], semantic, context, caches)
        return hidden

    def _layer(self, layer, index, hidden, codes, semantic, context, caches, shared_input=None):
        cache = caches[index]
        cache.prepare(hidden.shape[1])
        out = layer(hidden, codes[None], semantic[None], context, cache, shared_input[None] if shared_input is not None else None)
        cache.advance(hidden.shape[1])
        return out

    def step(self, main, hidden, frame, caches, sample):
        """Codes of the frames whose main tokens are `main` [B], after the frame rows `frame` [B, 1 + N] (one row per
        line decoded): the shared layers, then one residual layer and head per code, each code sampled before the
        next layer runs. Returns the frame rows [B, 1 + N]. `hidden` is [B, 1, H].

        The caller prepares the caches for one row before and advances them after, and the row's position is read
        from them on the GPU, so a CUDA graph can replay this."""
        context = self.context(caches[0].position[:len(main), None], 0, hidden.device, hidden.dtype)
        c0 = (main - self.semantic_start).clamp(0, self.cfg.vocab_size - 1)
        semantic = self.is_semantic(main)[:, None]
        unseen = self.embedding_sum(frame, 1 + self.cfg.backbone_codebooks, self.cfg.num_codebooks, hidden.dtype)
        unseen = torch.where(self.is_semantic(frame[:, 0])[:, None], unseen, 0.0)
        for index, layer in enumerate(self.shared_layers):
            hidden = layer(hidden, c0[:, None], semantic, context, caches[index], unseen[:, None] if index == 0 else None)
        codes = [c0]
        for index, (layer, head) in enumerate(zip(self.residual_layers, self.heads)):
            hidden = layer(hidden, codes[-1][:, None], semantic, context, caches[len(self.shared_layers) + index])
            codes.append(sample(head(hidden[:, 0])))
        return torch.stack((main, *codes), dim=-1)


class Drama3Model(nn.Module):
    """Backbone, the audio rows of the LM head, and the codebook tower. With a second GPU, the backbone's last layers,
    its final norm, the head and the tower move into a `SecondGPU` module that has a patcher of its own."""

    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.model = Drama3Backbone(config, device=device, dtype=dtype, ops=operations)
        self.audio_head = operations.Linear(config.text.hidden_size, 1 + config.tower.vocab_size, bias=False, device=device, dtype=dtype)
        self.tower = Drama3Tower(config, device=device, dtype=dtype, ops=operations)
        # run every frame, so made resident first; `SecondGPU` takes them along
        self.every_frame = [self.audio_head, self.tower]
        # loaded last: an evicted embedding table is read row by row from its host copy
        self.evictable = [self.model.embed_tokens]

    def get_dynamic_vram__units(self):
        return [*self.every_frame, *self.model.layers], self.evictable


class SecondGPU(nn.Module):
    """The modules a second GPU runs: the backbone layers from `first` on, the final norm, the head and the tower."""

    def __init__(self, model, first):
        super().__init__()
        self.first = first
        self.layers = nn.ModuleList(model.model.layers[first:])
        self.norm = model.model.norm
        self.audio_head = model.audio_head
        self.tower = model.tower
        model.model.layers = nn.ModuleList(model.model.layers[:first])
        del model.model.norm, model.audio_head, model.tower
        model.every_frame = []

    def get_dynamic_vram__units(self):
        return [self.audio_head, self.tower, *self.layers], []
