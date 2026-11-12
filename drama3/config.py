"""Drama 3 configuration, read from the vendor `config.json` that the converter stores in each model file."""
from dataclasses import dataclass

from comfy.text_encoders.qwen35 import Qwen35Config

# The converted file keeps only the LM head rows a frame's main token is drawn from: end of speech, then the
# 4096 semantic tokens. Row 0 is end of speech, row 1 + c is semantic code c.
AUDIO_HEAD_KEY = "audio_head.weight"
# ...and, in files converted since the 2026-10-29 weights, the rows of the newline and <|audio_separator|> under this
# key: a dialogue's take says them between its lines. Older files use the rows shipped in fish/line_head.safetensors,
# which are the first release's.
LINE_HEAD_KEY = "line_head.weight"
LINE_HEAD_IDS = (198, 252196)

# Repetition-aware sampling of the main token, fixed by the training recipe: when the first draw repeats one of
# the last RAS_WINDOW main tokens, a second draw with these parameters is used instead.
RAS_WINDOW = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
MIN_TEMPERATURE = 1e-5


@dataclass
class TowerConfig:
    dim: int
    n_head: int
    n_local_heads: int
    head_dim: int
    intermediate_size: int
    norm_eps: float
    num_codebooks: int
    vocab_size: int
    shared_layers: int
    backbone_codebooks: int
    rope_base: float
    rotary_dim: int


@dataclass
class Drama3Config:
    text: Qwen35Config
    tower: TowerConfig
    eos_token_id: int
    semantic_start_token_id: int
    semantic_end_token_id: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    # routed experts stored as the checkpoint's block FP8 (set from the converted file, not the vendor config)
    fp8_experts: bool = False


def parse_config(config):
    text = config["text_config"]
    if config.get("speech_template_version") != "reference-bracket-v1":
        raise ValueError(f"unsupported speech template {config.get('speech_template_version')!r}")
    if text["layer_types"] != ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(text["num_hidden_layers"])]:
        raise ValueError("unexpected Drama 3 layer layout")
    qwen = Qwen35Config(
        vocab_size=text["vocab_size"], hidden_size=text["hidden_size"], intermediate_size=text["moe_intermediate_size"],
        num_hidden_layers=text["num_hidden_layers"], num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"], head_dim=text["head_dim"],
        partial_rotary_factor=text["rope_parameters"]["partial_rotary_factor"],
        linear_num_key_heads=text["linear_num_key_heads"], linear_num_value_heads=text["linear_num_value_heads"],
        linear_key_head_dim=text["linear_key_head_dim"], linear_value_head_dim=text["linear_value_head_dim"],
        conv_kernel_size=text["linear_conv_kernel_dim"], max_position_embeddings=text["max_position_embeddings"],
        rms_norm_eps=text["rms_norm_eps"], rope_theta=text["rope_parameters"]["rope_theta"],
        layer_types=text["layer_types"], stop_tokens=[config["eos_token_id"]])
    decoder = config["audio_decoder_config"]
    tower = TowerConfig(
        dim=decoder["dim"], n_head=decoder["n_head"], n_local_heads=decoder["n_local_heads"], head_dim=decoder["head_dim"],
        intermediate_size=decoder["intermediate_size"], norm_eps=decoder["norm_eps"],
        num_codebooks=decoder["num_codebooks"], vocab_size=decoder["vocab_size"],
        shared_layers=decoder["shared_enrichment_layers"], backbone_codebooks=decoder["vq_backbone_codebooks"],
        rope_base=decoder["rope_base"], rotary_dim=int(decoder["head_dim"] * decoder["rotary_percent"]))
    return Drama3Config(
        text=qwen, tower=tower, eos_token_id=config["eos_token_id"],
        semantic_start_token_id=config["semantic_start_token_id"], semantic_end_token_id=config["semantic_end_token_id"],
        num_experts=text["num_experts"], num_experts_per_tok=text["num_experts_per_tok"],
        moe_intermediate_size=text["moe_intermediate_size"],
        shared_expert_intermediate_size=text["shared_expert_intermediate_size"])
