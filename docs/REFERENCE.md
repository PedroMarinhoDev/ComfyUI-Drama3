# ComfyUI-Drama3 reference

Everything the [README](../README.md) leaves out: the nodes and their settings, the script syntax in full, how the
model fits on a small card, the measurements behind the numbers, building the files yourself and how close the port
is to Fish Audio's code.

- [Nodes](#nodes)
- [Settings](#settings)
- [Script syntax](#script-syntax)
- [Dialogue modes](#dialogue-modes)
- [Files and requirements](#files-and-requirements)
- [Speed](#speed)
- [How it fits on an 8 GB card](#how-it-fits-on-an-8-gb-card)
- [Building the files yourself](#building-the-files-yourself)
- [How close it is to the official code](#how-close-it-is-to-the-official-code)
- [More troubleshooting](#more-troubleshooting)

## Nodes

| Node | What it does |
|---|---|
| **Load Drama 3** | Loads a model file. `experts`: in RAM (default) or in VRAM; `expert_cache_gb`: VRAM for cached experts (0 = what is free); `pin_experts`: keep the experts in pinned system memory, on by default and off on an integrated GPU; advanced: a second GPU for the experts-in-VRAM split. |
| **Load Drama 3 Codec** | Loads the codec from `models/vae`. |
| **Load Reference Audio** | Load Audio for a voice to clone. `trim` (on) cuts the clip's silent edges and keeps at most `length` (5–30 s, 20 s by default), cut in its latest pause before that. Fish Audio recommends 10–30 s of reference audio. |
| **Drama 3 Encode Voice** | A voice to clone: an audio clip and its exact transcript. |
| **Save Drama 3 Voice** / **Load Drama 3 Voice** | Keep an encoded voice (a few KB) in `output/drama3_voices` and use it again without the codec. |
| **Drama 3 Speed Options** | Between Load Drama 3 and Generate: the speed-ups that change what a seed generates, off by default. `lines_at_once` decodes several lines of a one-take-per-line script side by side. `repeatable_takes` keeps int8 experts off the CPU, so a seed gives the same take on every run, at the slower speed. |
| **Drama 3 Generate** | Text and any number of voices to audio codes. Long text is generated in chunks, each continuing from the first and the latest chunk. `voice_guidance` (0, off, by default) keeps the voices of a dialogue apart in the two modes where lines hear each other. |
| **Save Drama 3 Codes** / **Load Drama 3 Codes** | Keep a generated take (a few hundred KB) in `output/drama3_codes` and decode or edit it again without generating it again. |
| **Drama 3 Decode** | Codes to 44.1 kHz audio. |
| **Drama 3 Editor** | Codes to audio on a timeline, with one track per voice and captions that follow the words (with Qwen3-ForcedAligner on `aligner`). *Edit* opens the editor: set how the silences sound, check the lines it flags, cut, add silence, name the voices and set their volume. *Use these edits* runs the node again with them. |
| **Load Qwen3-ASR** | Loads a Qwen3-ASR or Qwen3-ForcedAligner file from `models/asr`. |
| **Qwen3-ASR Transcribe** | Audio to text and its language; `language` detects it or forces one of 30; `context` takes names and terms that help recognition. |

The codec runs only before and after generation, so it never sits in VRAM next to the model.

## Settings

- **temperature 1.0, top_p 0.9, top_k 0** (no cap) are the defaults of Fish Audio's Drama 3 Space. Lower temperature
  or top_p make takes slower, flatter and more likely to fall silent; top_k 30 makes takes fall silent at pauses.
- The model copies the reference's delivery: a flat clip gives a flat take, a lively one a lively take.
- **max_frames** caps each chunk (21.5 frames per second); generation stops earlier at the end of speech. 0, the
  default, sizes the cap from the chunk's text, so slow lines are not cut.
- Pause cues are riskier in a solo take: a line that opens with `[long pause]` can stay silent.
- As in Fish Audio's Space: **Encode Voice** trims the clip's silent edges and normalizes its peak; **Generate** takes
  a chunk once more when it has no sound or stalls in its first 2.5 s; **Decode**'s `shorten_long_pauses` (on) shortens
  pauses over 0.75 s to 0.75 s.
- A take that goes silent is stopped and its silent tail cut, with a warning in the console.
- **voice_guidance** (0, off, by default; 1.0 is a good value) works in **together** and **one take per line, hearing
  the line before**, with a voice for each speaker: each take is decoded beside the same text said in the other
  speaker's voice and steered away from it. On a two-voice dialogue (6 seeds), the lead of the second voice's lines
  (WavLM similarity to their own speaker's clip minus the other's) went from +0.02 to +0.08 hearing the line before
  and from +0.01 to +0.05 together, and none of 18 lines came out nearer the other voice, against 3 and 9 without it.
  2.0 gained little more and cost words in two takes. Those takes render about twice as long, a seed's takes change,
  and it needs one GPU and ComfyUI after v0.41.2; otherwise Generate warns and runs without it.

## Script syntax

Directions are free natural language, not a fixed list. The full syntax is Appendix C of Fish Audio's
[technical report](https://github.com/fishaudio/fish-speech/blob/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf).

| Control | How to write it | Example |
|---|---|---|
| Speaker | `<\|speaker:N\|>` at the start of a line, N from 0 to 15 | `<\|speaker:0\|> Ready?` |
| Point cue | `[cue]` where the sound should land | `Well [sighs] fine, you win. [laughs]` |
| Scoped cue | `<cue>...</cue>` around the words it shapes | `Promise me <whisper>you won't tell her</whisper>.` |
| Style | `[style]` at the start of a line | `[with bright eager excitement] We won!` |
| Director's note | `Direction:` with the note, then `Line:` with the text | `Direction: icy calm.` `Line: So this is goodbye.` |
| Language hint | `[language: xx]` after a line; it is not spoken | `Welcome back to the show. [language: en]` |

Text in round parentheses is spoken. Long text is cut into chunks of about 300 bytes without splitting a cue; keep a
director's note in one chunk with its line, or turn off `split_long_text`.

## Dialogue modes

Give every speaker a voice: the voice in Generate's slot `voiceN` speaks for `<|speaker:N|>`. A speaker without a
voice gets one the model makes up.

| `speakers` | How it generates | Voices stay apart | Lines react to each other |
|---|---|---|---|
| **one take per line** (default) | Each line on its own, hearing only its own speaker's earlier takes | yes | no |
| **one take per line, hearing the line before** | The same, with the take before it as context | the second voice can drift toward the first, unless guided | yes |
| **one take per speaker** | All of a speaker's lines in one take, cut at the line breaks and put back in script order | yes | no |
| **together** | One script in which every line hears the others | voices can blend, unless guided | yes |

In those two modes `voice_guidance` 1.0 keeps the voices apart ([Settings](#settings)), at about twice the render time
for those takes.

Separate takes are joined with 0.3 s of silence. `lines_at_once` on **Drama 3 Speed Options** changes what a seed
generates: lines decoded side by side cannot hear each other, so a line hears its speaker's takes that had finished
when it started.

## Files and requirements

From [Drama-3-ComfyUI](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI) and
[Qwen3-ASR-ComfyUI](https://huggingface.co/PedroMarinhoDev/Qwen3-ASR-ComfyUI) on Hugging Face:

| File | Goes in | Size |
|---|---|---|
| one of [`drama3_fp8experts_bf16`](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/blob/main/drama3_fp8experts_bf16.safetensors), [`drama3_int8experts_bf16`](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/blob/main/drama3_int8experts_bf16.safetensors), [`drama3_w6a8experts_bf16`](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/blob/main/drama3_w6a8experts_bf16.safetensors), [`drama3_w4a8experts_bf16`](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/blob/main/drama3_w4a8experts_bf16.safetensors) | `models/diffusion_models` | 24.6–38.7 GB |
| [`drama3_codec.safetensors`](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/blob/main/drama3_codec.safetensors) | `models/vae` | 1.6 GB |
| `qwen3_asr_1.7b_bf16.safetensors` | `models/asr` | 4.1 GB |
| `qwen3_asr_0.6b_bf16.safetensors` (optional: nearly the same transcripts on clean speech in half the time) | `models/asr` | 1.6 GB |
| `qwen3_forced_aligner_0.6b_bf16.safetensors` (optional: word timing in the editor) | `models/asr` | 1.8 GB |

| File | Experts | Distance from FP8 | Complete takes of 120 |
|---|---|---|---|
| [FP8](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/blob/main/drama3_fp8experts_bf16.safetensors) | as Fish Audio ships them | 1.1% (the floor) | 108 |
| [int8](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/blob/main/drama3_int8experts_bf16.safetensors) | int8 ConvRot, tuned | 1.2% | 109 |
| [W6A8](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/blob/main/drama3_w6a8experts_bf16.safetensors) | 6-bit weights, tuned | 1.6% | 108 |
| [W4A8](https://huggingface.co/PedroMarinhoDev/Drama-3-ComfyUI/blob/main/drama3_w4a8experts_bf16.safetensors) | 4-bit weights, tuned | 4.1% | 108 |

*Distance from FP8* compares the model's last hidden state over 2,391 held-out audio frames; the FP8 file through
plain float arithmetic scores 1.1% against itself, which is the floor. A take is *complete* when Qwen3-ASR hears at
least three quarters of its words, no silent tail was cut and the frame cap was not reached. The int8, W6A8 and W4A8
files are tuned on the model's own speech (GPTQ), not plain roundings.

| | Minimum | Tested |
|---|---|---|
| GPU | NVIDIA, 8 GB of VRAM | RTX 4090 and RTX 3090 (W4A8 2.7×, FP8 2.0× real time); 12 GB and 8 GB simulated by holding the rest of a card; both cards together |
| System RAM | the model file plus ~8 GB: 32 GB for W4A8, 48 GB for FP8 | 188 GB |
| Disk | 26 GB (W4A8 and the codec) | NVMe |
| ComfyUI | v0.40 | v0.41, v0.44 and v0.45 |
| CPU | any; with int8 on a small card more cores help | Intel Core i9-14900K |

## Speed

Drama 3 makes 21.5 audio frames per second of speech, so 46.5 ms per frame is real time. Measured on 2026-11-11 as
ComfyUI runs the nodes: a cloned voice, a 9.7 s line, the mean of two seeds, an RTX 4090 with an Intel Core i9-14900K.
*Before* is the pack as published on 2026-10-17 with the first release's files. The line's time includes the prompt
pass. The 12 GB and 8 GB rows ran with the rest of the card held by another process.

| GPU | File | Expert cache | ms per frame, before → now | 9.7 s line, before → now | Peak VRAM |
|---|---|---|---|---|---|
| RTX 4090, 24 GB | FP8 | automatic, 17.1 GiB | 23.7 → **17.4** | 7.7 → **6.5 s** | 23.1 GB |
| RTX 4090, 24 GB | int8 | automatic, 17.1 GiB | 22.4 → **17.1** | 6.9 → **6.6 s** | 22.9 GB |
| RTX 4090, 24 GB | W6A8 | automatic, 17.1 GiB | 18.1 → **14.6** | 6.2 → **5.7 s** | 23.1 GB |
| RTX 4090, 24 GB | W4A8 | automatic, 17.0 GiB | 12.9 → **12.1** | 4.6 → **4.6 s** | 22.7 GB |
| RTX 4090, 24 GB | FP8 | 4 GB | 58.4 → **50.7** | 14.1 → **12.8 s** | 9.8 GB |
| RTX 4090, 24 GB | int8 | 4 GB | 36.2 → **33.2** | 9.0 → **8.9 s** | 9.8 GB |
| RTX 4090, 24 GB | W6A8 | 4 GB | **39.9** | **10.8 s** | 9.8 GB |
| RTX 4090, 24 GB | W4A8 | 4 GB | 30.6 → **25.9** | 7.9 → **7.3 s** | 9.8 GB |
| RTX 4090 held to 12 GB | FP8 | automatic, 4.7 GiB | 67.1 → **47.1** | 15.7 → **12.1 s** | 10.9 GB |
| RTX 4090 held to 12 GB | int8 | automatic, 4.7 GiB | 39.6 → **31.2** | 9.8 → **8.5 s** | 11.1 GB |
| RTX 4090 held to 12 GB | W6A8 | automatic, 4.7 GiB | **36.5** | **10.1 s** | 11.1 GB |
| RTX 4090 held to 12 GB | W4A8 | automatic, 4.8 GiB | 36.8 → **23.3** | 9.1 → **6.7 s** | 10.9 GB |
| RTX 4090 held to 8 GB | FP8 | automatic, 0.7 GiB | 100.2 → **78.1** | 22.1 → **18.3 s** | 7.1 GB |
| RTX 4090 held to 8 GB | int8 | automatic, 0.7 GiB | 51.8 → **49.8** | 11.7 → **12.2 s** | 7.0 GB |
| RTX 4090 held to 8 GB | W6A8 | automatic, 0.7 GiB | **65.8** | **16.1 s** | 7.1 GB |
| RTX 4090 held to 8 GB | W4A8 | automatic, 0.8 GiB | 74.5 → **44.8** | 16.6 → **11.1 s** | 7.1 GB |
| RTX 3090, 24 GB | FP8 | 16 GiB | **22.7** | | 22.1 GB |
| RTX 3090, 24 GB | W4A8 | 16 GiB | **17.3** | | 22.1 GB |

Lines at once, RTX 4090:

| Use | File | 1 line at a time | 4 at once | 8 at once |
|---|---|---|---|---|
| Two-voice story, first ten minutes, 48 lines | W4A8 | 225 s | **111 s** | **107 s** |
| Two-voice story, first ten minutes, 48 lines | FP8 | 303 s | **199 s** | 229 s |
| Plain prose, 2,025 characters | W4A8 | 75.8 s | **32.7 s** | |
| 14 story lines, no voices given | W4A8 | 65.7 s | **43.1 s** | |
| Dialogue, two cloned voices, 7 lines | W4A8 | 13.5–14.1 s | **9.5–11.1 s** | |

W4A8's experts are the smallest, so more of them fit in a small cache and a miss costs less to read. With int8
experts and a small cache the CPU computes part of the experts the cache misses, so int8 holds up where FP8 and W6A8
slow down; a CPU with fewer or slower cores gets less of that help. Two cards sharing the model with the experts in
VRAM (`second_gpu`, an RTX 4090 and an RTX 3090, measured 2026-11-12) decode at 15.0 ms per frame, 3.1× real time,
with W4A8 and with FP8. Qwen3-ASR transcribes a 6.6 s clip in about 2 s with the 1.7B model and 1 s with
the 0.6B.

## How it fits on an 8 GB card

Drama 3 is a Qwen3.5-style mixture of experts: 40 layers, each with 256 routed experts of which 8 run per frame,
plus attention or DeltaNet, a shared expert and a 13-layer codebook decoder. Everything except the routed experts is
about 5 GB and stays on the GPU. With **experts in RAM** the experts stay in pinned system memory (`pin_experts` on
**Load Drama 3**) and the VRAM left free caches the most used ones (93% of FP8 or int8 reads hit a 17 GB cache, 99%
of W4A8's, which fits whole).

The pack's CUDA kernels, compiled at run time by PyTorch (no compiler or package to install), look each expert up in
the cache on the GPU, run the cached ones in place and read the missed ones straight from RAM, filling free cache
slots as they go. With int8 experts and a cache that holds under half of them (cards under about 20 GB) the CPU
computes part of the misses at the same time. Since nothing in a frame waits on the CPU, every layer and the codebook
decoder replay as CUDA graphs. Where PyTorch can't compile the kernels the nodes copy the experts instead, at the
first version's speed.

Pinning is the pack's own decision: `pin_experts` on **Load Drama 3** makes it, not ComfyUI's `--disable-pinned-memory`,
which speaks for ComfyUI's own weights. Pinned banks cost the same system RAM as unpinned ones, locked rather than
pageable, and they are copied a layer at a time, so the peak is that layer's banks above the file's size. With
`pin_experts` off every miss is copied to the VRAM instead, at the first version's speed, and `voice_guidance` and
`lines_at_once` stay off: neither runs in a frame that waits on the host.

With **experts in VRAM** the whole file sits on the GPU and the same kernels read the experts in place; two 24 GB
cards can split the layers (`second_gpu`).

## Building the files yourself

From the official checkpoints, one GPU visible (`CUDA_VISIBLE_DEVICES=N`), with ComfyUI on `PYTHONPATH`. These are
plain conversions: the int8, W6A8 and W4A8 files on Hugging Face were tuned after it (GPTQ on 66,000 rows of the
model's own takes, in the same layouts), which the pack does not ship.

```bash
python tools/convert.py model <drama-3 folder> drama3_fp8experts_bf16.safetensors --format fp8
python tools/convert.py model <drama-3 folder> drama3_int8experts_bf16.safetensors --format int8_convrot
python tools/convert.py model <drama-3 folder> drama3_w6a8experts_bf16.safetensors --format w6a8
python tools/convert.py model <drama-3 folder> drama3_w4a8experts_bf16.safetensors --format w4a8
python tools/convert.py codec <drama-3 folder>/codec.pth drama3_codec.safetensors
python tools/convert.py asr <Qwen3-ASR-1.7B folder> qwen3_asr_1.7b_bf16.safetensors
```

## How close it is to the official code

- **Drama 3**: against Fish Audio's own PyTorch code, with the same frames forced, the main token's distribution
  differs by a KL of 0.004–0.007 nats and the top token agrees 93–94% of the time, the noise of bf16 arithmetic.
  Prompt rows match the builders of Fish Audio's server and of fish-speech token for token, and the codec produces
  99.94% of the same codes.
- **Qwen3-ASR**: against Qwen's `qwen-asr` package, the prompt ids are identical, the encoder output matches within
  1e-6 in fp32 and the first token is the same; transcripts match on clips up to 8 s.

## More troubleshooting

**Speech is slow or flat.** Keep temperature at 1.0 and top_p at 0.9 or above, and use a reference with the delivery
you want.

**A clip keeps making takes fall silent.** Some clips make the model fall silent in Fish Audio's own code too; try
another stretch of the recording.
