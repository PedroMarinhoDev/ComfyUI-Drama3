"""Convert fishaudio/drama-3 to the files this pack loads.

    python tools/convert.py model <hf_dir> <out.safetensors> [--format fp8|int8_convrot|w6a8|w4a8] [--dense bf16|int8]
    python tools/convert.py codec <hf_dir>/codec.pth <out.safetensors>
    python tools/convert.py asr <Qwen3-ASR-1.7B, -0.6B or Qwen3-ForcedAligner-0.6B dir> <out.safetensors>

Model file: the backbone and the codebook tower under this pack's key names. The checkpoint ships its routed experts
as block-FP8 (128x128 scales); they are dequantized in fp32 and written as int8 ConvRot or W4A8 expert banks, the
layouts `comfy.ops.MoEExperts` loads. The other linears stay the checkpoint's bf16 or become int8 ConvRot (`--dense`).
Gate and up projections are fused (experts, shared expert, tower MLP). The LM head keeps only the rows generation
samples from: end of speech, then the 4096 semantic tokens; `line_head.weight` holds the rows of the newline and
<|audio_separator|>, which a dialogue's take says between its lines. Embeddings, heads, norms, the router and the small
DeltaNet gate projections stay bf16.

Codec file: the Fish S2-Pro codec with weight norm folded and the unused saved attention masks dropped, fp32.

Qwen3-ASR file: the checkpoint's bf16 tensors in one file, audio encoder under `audio_tower.` and decoder under
`decoder.model.`, without the LM head, which is the embedding table (checked); the config goes in the metadata.

Run with one visible GPU (CUDA_VISIBLE_DEVICES=N); the output is written to `<out>.part` and renamed when complete.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drama3.config import AUDIO_HEAD_KEY, LINE_HEAD_IDS, LINE_HEAD_KEY  # noqa: E402

CONVROT_GROUP_SIZE = 256
W4A8_GROUP_SIZE = 16
FP8_BLOCK = 128
LAYER_PREFIX = "model.language_model.layers."
BF16_SUFFIXES = (".in_proj_a.weight", ".in_proj_b.weight", ".mlp.gate.weight", ".shared_expert_gate.weight")
TOWER_PREFIXES = (("audio_decoder_mtp_shared.layers.", "tower.shared_layers."),
                  ("audio_decoder_mtp.layers.", "tower.residual_layers."),
                  ("audio_decoder_mtp.heads.", "tower.heads."))
TOWER_RENAMES = (("semantic_embedding.", "embedding."), ("codebook_embedding.", "embedding."),
                 ("transformer_layer.", ""))


def int8_convrot(weight, device):
    """[out, in] float -> (int8 [out, in], fp32 scale [out, 1]), rotated in fp32."""
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    qdata, params = TensorWiseINT8Layout.quantize(weight.to(device, torch.float32), per_channel=True, convrot=True,
                                                  convrot_groupsize=CONVROT_GROUP_SIZE)
    return qdata.cpu(), params.scale.float().cpu()


def quant_conf(**extra):
    conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE, **extra}
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def w4a8_bank(weights, device, bits=4):
    """Experts [out, in] -> W4A8 (or uniform 6-bit W6A8) bank tensors. Every expert uses the first one's codebook:
    a bank holds one table."""
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout
    qdatas, params, table = [], [], None
    for weight in weights:
        qdata, param = AsymW4A8Int8Layout.quantize(weight.to(device, torch.bfloat16), group_size=W4A8_GROUP_SIZE,
                                                    convrot_groupsize=CONVROT_GROUP_SIZE, codebook_tensor=table, bits=bits)
        if param.correction is not None:
            raise ValueError("W4A8 banks with a correction term are not supported")
        table = param.codebook
        qdatas.append(qdata.cpu())
        params.append(param)
    conf = {"format": "asym_w4a8_int8" if bits == 4 else "w6a8_int8", "group_size": W4A8_GROUP_SIZE,
            "convrot_groupsize": CONVROT_GROUP_SIZE, "num_experts": len(weights)}
    entries = {"weight": torch.stack(qdatas),
               "weight_s_rel": torch.stack([p.scale.cpu() for p in params]).view(torch.uint8),
               "weight_s_channel": torch.stack([p.s_channel.float().cpu() for p in params]),
               "comfy_quant": torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)}
    if table is not None:
        entries["weight_codebook"] = table.cpu()
    return entries, params, qdatas


def put_dense(out, module, weight, device, quantize):
    """A dense linear: int8 ConvRot, or the checkpoint's bf16 as is."""
    if not quantize:
        out[f"{module}.weight"] = weight.to(torch.bfloat16)
        return
    qdata, scale = int8_convrot(weight, device)
    out[f"{module}.weight"] = qdata
    out[f"{module}.weight_scale"] = scale
    out[f"{module}.comfy_quant"] = quant_conf()


def dequantize_fp8(weight, scale_inv, device):
    rows, cols = weight.shape
    blocks = weight.to(device).float().view(rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK)
    return (blocks * scale_inv.to(device)[:, None, :, None]).view(rows, cols)


def relative_error(qdata, scale, reference):
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    from comfy_kitchen.tensor.base import QuantizedTensor
    params = TensorWiseINT8Layout.Params(scale=scale.to(reference.device), orig_dtype=torch.float32,
                                         orig_shape=tuple(reference.shape), convrot=True,
                                         convrot_groupsize=CONVROT_GROUP_SIZE)
    back = QuantizedTensor(qdata.to(reference.device), "TensorWiseINT8Layout", params).dequantize().float()
    return float((back - reference).norm() / reference.norm())


class Checkpoint:
    def __init__(self, hf_dir):
        self.dir = Path(hf_dir)
        self.index = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def get(self, name):
        shard = self.index[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(self.dir / shard, framework="pt", device="cpu")
        return self.handles[shard].get_tensor(name)


def convert_model(args):
    config = json.loads((Path(args.src) / "config.json").read_text())
    text, tower_cfg = config["text_config"], config["audio_decoder_config"]
    device = torch.device(args.device)
    ckpt = Checkpoint(args.src)
    out = {}
    started = time.time()
    errors = []

    for name in ckpt.index:
        if name.startswith(LAYER_PREFIX) and ".mlp.experts." in name:
            continue
        if name == "lm_head.weight":
            head = ckpt.get(name)
            out[AUDIO_HEAD_KEY] = torch.cat((head[config["eos_token_id"]:config["eos_token_id"] + 1],
                                             head[config["semantic_start_token_id"]:config["semantic_end_token_id"] + 1])).clone()
            out[LINE_HEAD_KEY] = head[list(LINE_HEAD_IDS)].clone()
            continue
        if name.startswith("model.language_model."):
            key = "model." + name[len("model.language_model."):]
            if ".shared_expert.up_proj." in key or ".shared_expert.down_proj." in key:
                continue
            if ".shared_expert.gate_proj." in key:
                prefix = name.rsplit(".gate_proj.", 1)[0]
                fused = torch.cat((ckpt.get(f"{prefix}.gate_proj.weight"), ckpt.get(f"{prefix}.up_proj.weight")))
                module = key.rsplit(".gate_proj.", 1)[0]
                put_dense(out, f"{module}.gate_up_proj", fused, device, args.dense == "int8")
                put_dense(out, f"{module}.down_proj", ckpt.get(f"{prefix}.down_proj.weight"), device, args.dense == "int8")
                continue
            tensor = ckpt.get(name)
            if tensor.ndim == 2 and key.endswith(".weight") and "embed_tokens" not in key and not key.endswith(BF16_SUFFIXES):
                put_dense(out, key[:-len(".weight")], tensor, device, args.dense == "int8")
            else:
                out[key] = tensor
            continue
        if name == "audio_decoder.codebook_embeddings.weight":
            out["tower.codebook_embeddings.weight"] = ckpt.get(name)
            continue
        source, target = next((s, t) for s, t in TOWER_PREFIXES if name.startswith(s))
        key = target + name[len(source):]
        for old, new in TOWER_RENAMES:
            key = key.replace(old, new)
        if ".feed_forward.w3." in key:
            continue
        tensor = ckpt.get(name)
        if ".feed_forward.w1." in key:
            tensor = torch.cat((tensor, ckpt.get(name.replace(".w1.", ".w3."))))
            key = key.replace(".w1.", ".w13.")
        if tensor.ndim == 2 and key.startswith(("tower.shared_layers.", "tower.residual_layers.")) and not key.endswith(("embedding.weight", "norm.weight")):
            put_dense(out, key[:-len(".weight")], tensor, device, args.dense == "int8")
        else:
            out[key] = tensor
    print(f"dense tensors done in {time.time() - started:.0f}s", flush=True)

    experts = text["num_experts"]
    for layer in range(text["num_hidden_layers"]):
        if args.format == "fp8":
            # the checkpoint's own bytes and block scales, gate and up fused: nothing is rounded again
            prefix = f"{LAYER_PREFIX}{layer}.mlp.experts"
            for bank, projections in (("gate_up", ("gate_proj", "up_proj")), ("down", ("down_proj",))):
                for field, suffix in (("weight", "weight"), ("scale", "weight_scale_inv")):
                    out[f"model.layers.{layer}.mlp.experts.{bank}_{field}"] = torch.stack(
                        [torch.cat([ckpt.get(f"{prefix}.{e}.{proj}.{suffix}") for proj in projections]) for e in range(experts)])
            print(f"layer {layer}: experts copied, {time.time() - started:.0f}s", flush=True)
            continue
        banks = {"gate_up_proj": [], "down_proj": []}
        for expert in range(experts):
            prefix = f"{LAYER_PREFIX}{layer}.mlp.experts.{expert}"
            fp8 = {proj: dequantize_fp8(ckpt.get(f"{prefix}.{proj}.weight"), ckpt.get(f"{prefix}.{proj}.weight_scale_inv"), device)
                   for proj in ("gate_proj", "up_proj", "down_proj")}
            banks["gate_up_proj"].append(torch.cat((fp8["gate_proj"], fp8["up_proj"])))
            banks["down_proj"].append(fp8["down_proj"])
        for bank, weights in banks.items():
            module = f"model.layers.{layer}.mlp.experts.{bank}"
            if args.format in ("w4a8", "w6a8"):
                entries, params, qdatas = w4a8_bank(weights, device, 4 if args.format == "w4a8" else 6)
                if bank == "gate_up_proj":
                    from comfy_kitchen.tensor.base import QuantizedTensor
                    back = QuantizedTensor(qdatas[0].to(device), "AsymW4A8Int8Layout", params[0]).dequantize().float()
                    errors.append(float((back - weights[0]).norm() / weights[0].norm()))
                for suffix, tensor in entries.items():
                    out[f"{module}.{suffix}"] = tensor
            else:
                quantized = [int8_convrot(weight, device) for weight in weights]
                if bank == "gate_up_proj":
                    errors.append(relative_error(*quantized[0], weights[0]))
                out[f"{module}.weight"] = torch.stack([q for q, _ in quantized])
                out[f"{module}.weight_scale"] = torch.stack([scale for _, scale in quantized])
                out[f"{module}.comfy_quant"] = quant_conf(num_experts=experts)
            del weights
        print(f"layer {layer}: experts done, {time.time() - started:.0f}s, rel error {errors[-1]:.4f}", flush=True)

    metadata = {"drama3_format": args.format, "drama3_dense": args.dense, "drama3_config": json.dumps(config)}
    if errors:
        metadata["expert_rel_error_mean"] = f"{sum(errors) / len(errors):.5f}"
    write(out, args.out, metadata)


def fold_weight_norm(state):
    """g * v / |v| (norm over every dim but 0) for both weight-norm spellings the codec checkpoint uses."""
    folded = {}
    for key, tensor in state.items():
        if key.endswith(("freqs_cis", "causal_mask")):
            continue
        for g_name, v_name, weight_name in ((".weight_g", ".weight_v", ".weight"),
                                            (".parametrizations.weight.original0", ".parametrizations.weight.original1", ".weight")):
            if key.endswith(g_name):
                base = key[:-len(g_name)]
                folded[base + weight_name] = torch._weight_norm(state[base + v_name].float(), tensor.float(), 0)
                break
            if key.endswith(v_name):
                break
        else:
            folded[key] = tensor.float() if tensor.is_floating_point() else tensor
    return folded


def convert_codec(args):
    state = torch.load(args.src, map_location="cpu", weights_only=True)
    write(fold_weight_norm(state), args.out, {"drama3_format": "codec"})


def convert_asr(args):
    src = Path(args.src)
    root = json.loads((src / "config.json").read_text())
    config = root["thinker_config"]
    index = src / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    prefixes = (("thinker.audio_tower.", "audio_tower."), ("thinker.model.", "decoder.model."))
    out, head = {}, None
    for shard in shards:
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name == "thinker.lm_head.weight":
                    head = f.get_tensor(name)
                    continue
                source, target = next((s, t) for s, t in prefixes if name.startswith(s))
                out[target + name[len(source):]] = f.get_tensor(name)
    kept = {key: config[key] for key in ("audio_config", "text_config")}
    if "classify_num" in config:
        # Qwen3-ForcedAligner: its head classifies timestamps in steps of timestamp_segment_time ms
        out["timestamp_head.weight"] = head
        kept.update(classify_num=config["classify_num"], timestamp_token_id=root["timestamp_token_id"],
                    timestamp_segment_time=root["timestamp_segment_time"])
    elif head is not None and not torch.equal(head, out["decoder.model.embed_tokens.weight"]):
        raise ValueError("this checkpoint's LM head is not its embedding table")
    write(out, args.out, {"qwen3_asr_config": json.dumps(kept)})


def write(tensors, path, metadata):
    part = f"{path}.part"
    save_file({k: v.contiguous() for k, v in tensors.items()}, part, metadata=metadata)
    os.replace(part, path)
    size = os.path.getsize(path)
    print(f"wrote {path}: {len(tensors)} tensors, {size / 1e9:.2f} GB", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("model", "codec", "asr"))
    parser.add_argument("src")
    parser.add_argument("out")
    parser.add_argument("--format", default="int8_convrot", choices=("fp8", "int8_convrot", "w6a8", "w4a8"),
                        help="routed experts: the checkpoint's block FP8 as is (32.2 GB), int8 ConvRot (32.2 GB), W6A8 "
                             "(26.2 GB) or W4A8 (18.1 GB)")
    parser.add_argument("--dense", default="bf16", choices=("int8", "bf16"),
                        help="every linear but the routed experts (attention, DeltaNet, shared expert, tower): the checkpoint's "
                             "bf16 (5.3 GB, fastest at one frame per step) or int8 ConvRot (2.7 GB)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists")
    {"model": convert_model, "codec": convert_codec, "asr": convert_asr}[args.kind](args)


if __name__ == "__main__":
    main()
