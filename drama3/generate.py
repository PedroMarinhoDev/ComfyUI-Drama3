"""Autoregressive frame generation: prefill the prompt rows, then per frame sample the main token from the audio
head, the other codes from the tower, and run the backbone on the new frame."""
import concurrent.futures
import logging

import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
from .config import MIN_TEMPERATURE, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, RAS_WINDOW
from .model import DEFERRED_DECODE, LinearKV

# A take that falls silent without ending repeats a handful of codes: 96 frames (4.5 s) of speech hold 80-95
# distinct first codes and the most expressive measured stretch (sighs, laughs) 41; silent runaways 5-20.
STALL_WINDOW = 96
STALL_DISTINCT = 24
STALL_TAIL = 10
# Fish Audio's Space holds back a take's first 2.5 s and retries it once if they never come out.
OPENING_FRAMES = 54
OPENING_ATTEMPTS = 2
# Lines decoded together: a line loaded into a row starts from the state kept after the rows its prompt shares with
# an earlier one (a speaker's voice and first take), when they share this many more than a kept state covers; its own
# rows after them stay more than a decode step takes (ComfyUI: 8). The latest prompts are looked at for shared rows.
SHARED_ROWS = 64
OWN_ROWS = 16
RECENT_PROMPTS = 8


def sample_top_k_top_p(logits, temperature, top_k, top_p, generator):
    """One id per row; drops the token crossing top_p except the highest-ranked one (the vendor's rule). top_k 0 keeps
    every id."""
    values, ids = torch.sort(logits.float() / max(temperature, MIN_TEMPERATURE), descending=True, dim=-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= (top_k or values.shape[-1]))
    dropped[..., 0] = False
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    noise = torch.empty_like(probabilities).exponential_(generator=generator)
    return ids.gather(-1, (probabilities / noise).argmax(-1, keepdim=True)).squeeze(-1)


def embed_rows(embed, ids, dtype):
    """Token embeddings; an evicted dynamic table gathers the rows from its host copy instead of streaming whole."""
    if (hasattr(embed, "_v") and embed.weight_lowvram_function is None and len(embed.weight_function) == 0
            and comfy.ops.vbar_above_watermark(embed)):
        return embed.host_rows(ids, out_dtype=dtype)
    return embed(ids, out_dtype=dtype)


def generate(drama3, prompt, max_frames, temperature, top_k, top_p, seed, on_frame=None, lines=False, contrast=None, guidance=0.0):
    """Frames [frames, 1 + N] (CPU int64) sampled after the prompt rows [rows, 1 + N], without the end frame.

    With `lines` (a script whose lines are joined by <|audio_separator|>) the take may also say the newline and the
    <|audio_separator|> that close each line's audio, as Drama 3 was trained to (report, Figure 11); they come back as
    text rows [id, 0, ...] between the lines' audio frames.

    A take without sound, or one that stalls within its first OPENING_FRAMES, is taken once more, drawing on from the
    same generator, as Fish Audio's Drama 3 Space retries a silent or incomplete opening.

    With `contrast` (the prompt rows of the same text in another voice; needs `batchable`) the take is decoded beside
    it, each step drawing from the logits pushed away from the contrast's: l + guidance * (l - l_contrast)."""
    config = drama3.config
    generator = torch.Generator(device=drama3.parts()[1][-1]).manual_seed(seed)
    for attempt in range(OPENING_ATTEMPTS):
        if contrast is None:
            frames, stalled = take(drama3, prompt, max_frames, temperature, top_k, top_p, generator, on_frame, lines)
        else:
            pair, out = [("line", prompt, max_frames), ("contrast", contrast, max_frames)], {}
            decode_lines(drama3, 2, max(len(prompt), len(contrast)) + max_frames, max_frames, lambda: pair.pop(0) if pair else None,
                         lambda line, frames, stalled: out.setdefault(line[0], (frames, stalled)), temperature, top_k, top_p,
                         generator, on_frame, guidance, scripts=lines)
            frames, stalled = out["line"]
        sound = int(((frames[:, 0] >= config.semantic_start_token_id) & (frames[:, 0] <= config.semantic_end_token_id)).sum())
        if sound >= OPENING_FRAMES or (sound and not stalled):
            break
        if attempt + 1 < OPENING_ATTEMPTS:
            logging.warning("Drama 3: the take had no complete opening; taking it once more.")
    if stalled:
        logging.warning("Drama 3: the take went silent without ending after %.1f s; its silent tail was cut. "
                        "Another seed or a voice reference usually fixes this.", sound / 21.5)
    return frames


def batchable(drama3):
    """Whether several lines can be decoded in one batch, inside a session: one device, ComfyUI's decode after v0.41.2,
    the layers replayed as CUDA graphs, and every expert read by the kernels on the GPU."""
    layers, devices, _, _, _ = drama3.parts()
    return (DEFERRED_DECODE and drama3.second is None and comfy.model_prefetch.malloc_graph_enabled(devices[0])
            and all(layer.mlp.experts.sync_free(devices[0]) for layer in layers)
            and (drama3.experts is None or drama3.experts.cpu_side is None))


def session_rows(capacity, width):
    """The rows to open a session with for `width` lines decoded together, each of at most `capacity` rows: a line's
    caches take a sixth of what a prompt's pass does per row (50 KiB of 0.3 MiB), the kept prompt states at most one
    line's."""
    return capacity * (5 + width + (width > 1)) // 6


def generate_lines(drama3, width, capacity, most, next_line, keep, temperature, top_k, top_p, seed, on_frame=None):
    """`generate` for lines that do not hear each other, `width` of them decoded together from one generator; a line
    that ends leaves its row to the next. `next_line()` gives (key, prompt rows, max frames) or None while no line may
    start; `keep(key, frames)` takes a finished line. A line without a complete opening is taken once more.
    `capacity`: the rows a line's caches hold (its prompt and frames), `most` the largest max frames. Needs
    `batchable`."""
    config = drama3.config
    generator = torch.Generator(device=drama3.parts()[1][0]).manual_seed(seed)
    again, attempts = [], {}

    def source():
        return again.pop(0) if again else next_line()

    def finished(line, frames, stalled):
        key = line[0]
        sound = int(((frames[:, 0] >= config.semantic_start_token_id) & (frames[:, 0] <= config.semantic_end_token_id)).sum())
        attempts[key] = attempts.get(key, 0) + 1
        if sound < OPENING_FRAMES and (stalled or not sound) and attempts[key] < OPENING_ATTEMPTS:
            logging.warning("Drama 3: the take had no complete opening; taking it once more.")
            again.append(line)
            return
        if stalled:
            logging.warning("Drama 3: the take went silent without ending after %.1f s; its silent tail was cut. "
                            "Another seed or a voice reference usually fixes this.", sound / 21.5)
        keep(key, frames)

    decode_lines(drama3, width, capacity, most, source, finished, temperature, top_k, top_p, generator, on_frame)


def decode_lines(drama3, width, capacity, most, source, finished, temperature, top_k, top_p, generator, on_frame=None, guidance=0.0,
                 scripts=False):
    """Lines decoded `width` at a time, a frame of each per step. `source()` gives the next line (key, prompt rows, max
    frames) or None while none may start; `finished(line, frames, whether it stalled and its silent tail was cut)`
    takes each as it ends, and its row goes to the next line. `scripts`: the lines are scripts, as with `lines` in
    `generate`.

    With `guidance` the lines come in pairs, a line and its contrast (the same text in another voice): both rows take
    the tokens drawn from l + guidance * (l - l_contrast), so they end together."""
    config = drama3.config
    backbone, dtype = drama3.model.model, drama3.model.dtype
    layers, devices, norm, audio_head, tower = drama3.parts()
    device = devices[0]
    candidates = torch.cat((torch.tensor([config.eos_token_id], device=device),
                            torch.arange(config.semantic_start_token_id, config.semantic_end_token_id + 1, device=device)))
    line_head = None
    if scripts:
        line_ids, line_head = drama3.line_head
        candidates = torch.cat((candidates, line_ids.to(device)))
        line_head = comfy.model_management.cast_to(line_head, dtype, device)

    def embed(frames):
        return tower.embed_backbone(frames, embed_rows(backbone.embed_tokens, frames[:, 0], dtype))

    def sample(logits, temperature=temperature, top_k=top_k, top_p=top_p):
        if guidance:
            logits = torch.lerp(logits[0::2].float(), logits[1::2].float(), -guidance)
        tokens = sample_top_k_top_p(logits, temperature, top_k, top_p, generator)
        return tokens.repeat_interleave(2) if guidance else tokens

    with drama3.session(session_rows(capacity, width)), comfy.model_management.cuda_device_context(device):
        caches, side = backbone.init_batch_caches(width, capacity, layers, device, dtype)
        tower_caches = tower.init_batch_caches(width, capacity, device, dtype)
        codebooks = 1 + config.tower.num_codebooks
        # what carries from frame to frame stays at fixed addresses, as in `take`
        tower_io = {"main": torch.empty((width,), dtype=torch.long, device=device),
                    "hidden": torch.zeros((width, 1, backbone.config.hidden_size), dtype=dtype, device=device),
                    "frame": torch.zeros((width, codebooks), dtype=torch.long, device=device),
                    "out": torch.empty((width, codebooks), dtype=torch.long, device=device)}
        history = torch.full((width, RAS_WINDOW), -1, dtype=torch.long, device=device)
        # each row's frames, the step its line started at, and the position of its line's first frame less that step
        taken = torch.zeros((width, most, codebooks), dtype=torch.long, device=device)
        since = torch.zeros((width,), dtype=torch.long, device=device)
        base = torch.zeros((width, 1), dtype=torch.long, device=device)
        rows_index = torch.arange(width, device=device)
        hidden_buffer, rotary = backbone.init_decode_buffers(width, device, dtype)
        decode_buffers = (hidden_buffer, rotary.repeat(width, *[1] * (rotary.dim() - 1)))
        lines, started, codes = [None] * width, [0] * width, [[] for _ in range(width)]
        # states kept after the rows prompts share (the rows, the last one's hidden state, what the caches and the
        # tower's hold), and the latest prompts, to find what a new one shares
        kept, recent = [], []

        def held(cache, rows):
            """What a cache holds after `rows` rows: a DeltaNet layer's whole state, an attention layer's rows."""
            if isinstance(cache, LinearKV):
                return cache.key, cache.value
            return cache.key[:, :, :rows], cache.value[:, :, :rows]

        def common(a, b, limit):
            """How many of their first `limit` rows two prompts share."""
            limit = min(limit, len(a), len(b))
            differ = (a[:limit] != b[:limit]).any(-1).nonzero()
            return int(differ[0]) if len(differ) else limit

        def run(own, own_tower, prompt, start, stop, before):
            """Rows [start, stop) of a prompt through a row's caches, which hold the rows before (`before`: the last
            of their hidden states), and the tower over the rows it can label: a row is labelled with the next one,
            so the last row waits for the call after, or for the decode step when `stop` leaves only that."""
            hidden = norm(backbone.run_layers(embed(prompt[start:stop])[None], own, layers, devices))
            states = hidden if stop == len(prompt) - 1 else hidden[:, :-1]
            if before is not None:
                states = torch.cat((before, states), dim=1)
            first, count = start - (before is not None), states.shape[1]
            tower.run(states, prompt[first + 1:first + 1 + count], prompt[first:first + count],
                      tower.context(torch.arange(first, first + count, device=device), count, device, dtype), own_tower)
            return hidden[:, -1:]

        def load(row, line, step):
            """A line into a free row, between steps: its prompt pass, from a kept state where its prompt begins as an
            earlier one did, and its last prompt row as a decode step whose token the batch's next step commits."""
            rows = line[1]
            limit = max(len(rows) - 1 - OWN_ROWS, 0)
            own, own_tower = backbone.row_caches(caches, row, side), tower.row_caches(tower_caches, row)
            start, before = 0, None
            for state in sorted(kept, key=lambda state: -len(state[0])):
                if len(state[0]) <= limit and common(rows, state[0], limit) == len(state[0]):
                    start, before = len(state[0]), state[1]
                    for stack, saved, count in ((own, state[2], start), (own_tower, state[3], start - 1)):
                        for cache, tensors in zip(stack, saved):
                            for target, source in zip(held(cache, count), tensors):
                                target.copy_(source)
                            cache.index = count
                    # the states least lately used go first
                    kept.sort(key=lambda other: other is state)
                    break
            shared = max((common(rows, earlier, limit) for earlier in recent), default=0)
            recent.append(rows)
            del recent[:-RECENT_PROMPTS]
            prompt = rows.to(device)
            if shared - start >= SHARED_ROWS:
                before = run(own, own_tower, prompt, start, shared, before)
                start = shared
                kept.append((rows[:start], before.clone(), [[tensor.clone() for tensor in held(cache, start)] for cache in own],
                             [[tensor.clone() for tensor in held(cache, start - 1)] for cache in own_tower]))
                while sum(len(state[0]) for state in kept) > capacity:
                    del kept[0]
            run(own, own_tower, prompt, start, len(prompt) - 1, before)
            tower_io["hidden"][row].copy_(norm(backbone.run_layers(embed(prompt[-1:])[None], own, layers, devices))[0])
            tower_io["frame"][row].copy_(prompt[-1])
            base[row] = backbone.adopt_row(caches, own, row) - step
            tower.adopt_row(tower_caches, own_tower, row)
            since[row] = step
            history[row].fill_(-1)
            history[row, RAS_WINDOW - min(RAS_WINDOW, len(prompt)):] = prompt[-RAS_WINDOW:, 0]
            lines[row], started[row], codes[row] = line, step, []

        def refill(step):
            for row in range(width):
                if lines[row] is None:
                    line = source()
                    if line is not None:
                        load(row, line, step)
                    else:
                        # an idle row still writes a token per step: from the start of its caches again
                        backbone.empty_row(caches, row)
                        tower.empty_row(tower_caches, row)

        def tower_core():
            tower_io["out"].copy_(tower.step(tower_io["main"], tower_io["hidden"], tower_io["frame"], tower_caches, sample))

        refill(0)
        step = 0
        try:
            while any(line is not None for line in lines):
                if step:
                    comfy.model_prefetch.malloc_graph_begin(device)
                logits = audio_head(tower_io["hidden"][:, 0])
                if line_head is not None:
                    logits = torch.cat((logits, torch.nn.functional.linear(tower_io["hidden"][:, 0], line_head)), dim=-1)
                draw = candidates[sample(logits)]
                retry = candidates[sample(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P)]
                repeated = (history == draw[:, None]).any(-1) & tower.is_semantic(draw)
                if guidance:
                    repeated = repeated[0::2].repeat_interleave(2)
                tower_io["main"].copy_(torch.where(repeated, retry, draw))
                for cache in tower_caches:
                    cache.prepare(1)
                queue = comfy.model_prefetch.make_prefetch_queue([tower], device, {"prefetch_dynamic_vbars": True})
                comfy.model_prefetch.prefetch_queue_pop(queue, device, tower, dtype, core=tower_core, enable_graph=True,
                                                        generator=generator, malloc_scope="tower")
                comfy.model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="tower")
                for cache in tower_caches:
                    cache.advance(1)
                frame = tower_io["out"]
                said = torch.cat((frame[:, 0], frame[:, 1], torch.isfinite(logits).all(-1).long())).tolist()
                del logits, draw, retry, repeated
                ended = []
                for row in range(width):
                    if lines[row] is None:
                        continue
                    if not said[2 * width + row]:
                        raise FloatingPointError("Drama 3 produced non-finite logits")
                    main_id, at, end, stalled = said[row], step - started[row], None, False
                    if main_id == config.eos_token_id:
                        end = at
                    elif not config.semantic_start_token_id <= main_id <= config.semantic_end_token_id:
                        frame[row, 1:] = 0
                    else:
                        codes[row].append(said[width + row])
                        if len(codes[row]) >= STALL_WINDOW and len(set(codes[row][-STALL_WINDOW:])) <= STALL_DISTINCT:
                            end, stalled = at + 1 - STALL_WINDOW + STALL_TAIL, True
                    if end is None and at + 1 == lines[row][2]:
                        end = at + 1
                        logging.warning("Drama 3: a chunk reached max_frames (%d) before its end of speech; its last words may be "
                                        "missing. Raise max_frames.", lines[row][2])
                    if end is not None:
                        ended.append((row, end, stalled))
                # an idle row keeps writing, within its own buffer
                taken[rows_index, (step - since).clamp_(max=most - 1)] = frame
                history.copy_(torch.cat((history[:, 1:], frame[:, :1]), dim=1))
                tower_io["frame"].copy_(frame)
                if on_frame is not None:
                    on_frame(step)
                tower_io["hidden"].copy_(norm(backbone.run_layers(embed(frame)[:, None], caches, layers, devices, decode_buffers,
                                                                  positions=base + step)))
                if step:
                    comfy.model_prefetch.malloc_graph_end()
                step += 1
                for row, end, stalled in ended:
                    line, lines[row] = lines[row], None
                    finished(line, taken[row, :end].cpu(), stalled)
                if ended:
                    refill(step)
        finally:
            # the graphs hold these caches
            comfy.model_prefetch.cleanup_prefetch_queues()


def take(drama3, prompt, max_frames, temperature, top_k, top_p, generator, on_frame=None, lines=False):
    """One attempt of `generate`: (frames, whether it stalled and its silent tail was cut)."""
    config = drama3.config
    backbone = drama3.model.model
    dtype = drama3.model.dtype
    layers, devices, norm, audio_head, tower = drama3.parts()
    first, last = devices[0], devices[-1]
    rows = prompt.to(last)
    prompt_rows = len(rows)
    capacity = prompt_rows + max_frames + 1
    caches = backbone.init_caches(capacity, layers, devices, dtype)
    tower_caches = tower.init_caches(capacity, last, dtype)
    candidates = torch.cat((torch.tensor([config.eos_token_id], device=last),
                            torch.arange(config.semantic_start_token_id, config.semantic_end_token_id + 1, device=last)))
    line_head = None
    if lines:
        line_ids, line_head = drama3.line_head
        candidates = torch.cat((candidates, line_ids.to(last)))
        line_head = comfy.model_management.cast_to(line_head, dtype, last)

    def embed(frames):
        text = embed_rows(backbone.embed_tokens, frames[:, 0].to(first), dtype)
        return tower.embed_backbone(frames, text)[None]

    def run(x, start, end, decode_buffers=None, after_layer=None, ahead=False):
        """Layers [start, end) on x; `after_layer` takes the layer's index in the whole stack."""
        return backbone.run_layers(x, caches[start:end], layers[start:end], devices[start:end], decode_buffers,
                                   after_layer and (lambda index: after_layer(start + index)), ahead)

    def sample(logits):
        return sample_top_k_top_p(logits, temperature, top_k, top_p, generator)

    frames, codes, stalled = [], [], False
    with drama3.session(capacity):
        with comfy.model_management.cuda_device_context(last):
            prefill = norm(run(embed(rows), 0, len(layers)))
        history = torch.full((RAS_WINDOW,), -1, dtype=torch.long, device=last)
        window = rows[-RAS_WINDOW:, 0]
        history[RAS_WINDOW - len(window):] = window
        # what carries from frame to frame stays at fixed addresses: the tower's CUDA graph reads and writes them, and
        # ComfyUI's model compiler, which records each frame's allocations, needs every other tensor freed by its end
        taken = torch.empty((max_frames, rows.shape[1]), dtype=torch.long, device=last)
        tower_io = {"main": torch.empty((1,), dtype=torch.long, device=last), "hidden": prefill[:, -1:].clone(),
                    "frame": rows[-1:].clone(), "out": torch.empty_like(rows[-1:])}
        hidden = tower_io["hidden"]

        def tower_core():
            tower_io["out"].copy_(tower.step(tower_io["main"], hidden, tower_io["frame"], tower_caches, sample))

        # the layers on the second GPU (all of them with one); a second GPU's first layers run in a thread of their own,
        # since ComfyUI's model compiler records one device's allocations per thread
        split = devices.index(last)
        compile = comfy.model_prefetch.malloc_graph_enabled(last)
        # the backbone's layers replay as CUDA graphs too when nothing in them waits on the host
        sync_free = all(layer.mlp.experts.sync_free(device) for layer, device in zip(layers, devices))
        decode_buffers = backbone.init_decode_buffers(1, last, dtype) if compile and sync_free else None
        first_buffers = None
        if split and sync_free and comfy.model_prefetch.malloc_graph_enabled(first):
            first_buffers = backbone.init_decode_buffers(1, first, dtype)
        first_half = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="drama3-first-gpu") if split else None
        # the experts a decode step leaves to the CPU, run between the layers
        after_layer = drama3.experts.after_layer if drama3.experts is not None and drama3.experts.cpu_side is not None else None

        # the thread runs as this one does: the caches and buffers it updates are this mode's tensors
        inference = torch.is_inference_mode_enabled()

        def first_layers(x, step, ahead):
            with torch.inference_mode(inference), comfy.model_management.cuda_device_context(first):
                graphs = first_buffers is not None and step > 0
                if graphs:
                    comfy.model_prefetch.malloc_graph_begin(first)
                x = run(x, 0, split, first_buffers, after_layer, ahead)
                if graphs:
                    comfy.model_prefetch.malloc_graph_end()
                return x
        with comfy.model_management.cuda_device_context(last):
            if prompt_rows > 1:
                count = prompt_rows - 1
                tower.run(prefill[:, :-1], rows[1:], rows[:-1], tower.context(torch.arange(count, device=last), count, last, dtype), tower_caches)
            del prefill
            x = None
            try:
                for step in range(max_frames):
                    if compile and step:
                        comfy.model_prefetch.malloc_graph_begin(last)
                    if x is not None:
                        # the second GPU's layers of the frame the first GPU's thread ran last (one recording at a time)
                        hidden.copy_(norm(run(x, split, len(layers), decode_buffers, after_layer, step > 3)))
                    logits = audio_head(hidden[0])
                    if line_head is not None:
                        logits = torch.cat((logits, torch.nn.functional.linear(hidden[0], line_head)), dim=-1)
                    draw = candidates[sample(logits)]
                    retry = candidates[sample_top_k_top_p(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, generator)]
                    repeated = (history == draw[:, None]).any(-1) & tower.is_semantic(draw)
                    tower_io["main"].copy_(torch.where(repeated, retry, draw))
                    for cache in tower_caches:
                        cache.prepare(1)
                    queue = comfy.model_prefetch.make_prefetch_queue([tower], last, {"prefetch_dynamic_vbars": True})
                    comfy.model_prefetch.prefetch_queue_pop(queue, last, tower, dtype, core=tower_core, enable_graph=True,
                                                            generator=generator, malloc_scope="tower")
                    comfy.model_prefetch.prefetch_queue_pop(queue, last, None, malloc_scope="tower")
                    if after_layer is not None:
                        drama3.experts.between_steps()
                    for cache in tower_caches:
                        cache.advance(1)
                    frame = taken[step:step + 1]
                    frame.copy_(tower_io["out"])
                    main_id, code, finite = torch.stack((frame[0, 0], frame[0, 1], torch.isfinite(logits).all().long())).tolist()
                    del logits, draw, retry, repeated
                    if not finite:
                        raise FloatingPointError("Drama 3 produced non-finite logits")
                    if main_id == config.eos_token_id:
                        break
                    if not config.semantic_start_token_id <= main_id <= config.semantic_end_token_id:
                        frame[:, 1:] = 0
                    else:
                        codes.append(code)
                    frames.append(frame)
                    if len(codes) >= STALL_WINDOW and len(set(codes[-STALL_WINDOW:])) <= STALL_DISTINCT:
                        del frames[len(frames) - STALL_WINDOW + STALL_TAIL:]
                        stalled = True
                        break
                    history.copy_(torch.cat((history[1:], frame[:, 0])))
                    tower_io["frame"].copy_(frame)
                    if on_frame is not None:
                        on_frame(step)
                    if step + 1 < max_frames and not split:
                        # the host serves a layer's CPU experts after launching the next one, once ComfyUI has recorded
                        # the layers' graphs (step 2): recording waits for the GPU, which would wait for the host
                        hidden.copy_(norm(run(embed(frame), 0, len(layers), decode_buffers, after_layer, step > 2)))
                    if compile and step:
                        comfy.model_prefetch.malloc_graph_end()
                    if step + 1 < max_frames and split:
                        # embedded here: the codebook embeddings sit with the tower, on the second GPU
                        x = first_half.submit(first_layers, embed(frame), step, step > 2).result()
                else:
                    logging.warning("Drama 3: a chunk reached max_frames (%d) before its end of speech; its last words may be "
                                    "missing. Raise max_frames.", max_frames)
            finally:
                # the graphs hold this take's caches
                if first_half is not None:
                    first_half.submit(comfy.model_prefetch.cleanup_malloc_graph).result()
                    first_half.shutdown()
                comfy.model_prefetch.cleanup_prefetch_queues()
    if not frames:
        return torch.zeros((0, rows.shape[1]), dtype=torch.long), stalled
    return torch.cat(frames).cpu(), stalled
