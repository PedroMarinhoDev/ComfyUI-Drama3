"""CUDA kernels for a layer's routed experts held in host memory, compiled at first use with NVRTC (no CUDA toolkit or
compiler needed).

The experts live in pinned host memory and a slot cache on the GPU. For a decode row, `resolve_experts` looks its
experts up in a table on the GPU and gives each missed expert the slot of the least used cached one; the GEMV kernels
then read a cached expert from its slot and a missed one straight from host memory, writing it into its new slot as
they go. Nothing waits on the host, so a decode step can be captured in a CUDA graph. The grouped kernels run a
prompt's rows the same way, one launch per bank.
"""
import functools

import torch

SOURCE = r"""
#define WARPS 8
#define ROWS_PER_WARP 2
#define ROWS_PER_BLOCK (WARPS * ROWS_PER_WARP)
#define INT_MAX 2147483647

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Float8 e4m3 to float, subnormals included: the 7 magnitude bits land in the float's exponent and mantissa, 2^120
// rebiases the exponent.
__device__ __forceinline__ float e4m3(unsigned int b) {
  return __int_as_float(((b & 0x80u) << 24) | ((b & 0x7Fu) << 20)) * __int_as_float(0x7b800000);
}

__device__ __forceinline__ float from_bf16(unsigned short b) { return __int_as_float(((unsigned int)b) << 16); }

// Float to bfloat16, to nearest and ties to even, as a cast does.
__device__ __forceinline__ unsigned short to_bf16(float v) {
  unsigned int b = (unsigned int)__float_as_int(v);
  return (unsigned short)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}

// A decode row's routing: the k largest of n logits (bfloat16) in descending order (ids) and their softmax weights
// renormalized over those k, each rounded to bfloat16 as the model's own cast rounds them. n <= 1024, k <= 32.
extern "C" __global__ void __launch_bounds__(256) route_experts(const unsigned short* logits, int n, int k,
                                                                long long* ids, float* weights) {
  __shared__ float value[1024];
  __shared__ float top[32];
  __shared__ int top_id[32];
  __shared__ float total;
  for (int i = threadIdx.x; i < n; i += blockDim.x) value[i] = from_bf16(logits[i]);
  if (threadIdx.x < 32) { top[threadIdx.x] = 0.0f; top_id[threadIdx.x] = 0; }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = value[i];
    int rank = 0;
    for (int j = 0; j < n; j++) rank += value[j] > v || (value[j] == v && j < i);
    if (rank < k) { top[rank] = v; top_id[rank] = i; }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int e = 0; e < k; e++) sum += expf(top[e] - top[0]);
    total = sum;
  }
  __syncthreads();
  if (threadIdx.x < k) {
    weights[threadIdx.x] = from_bf16(to_bf16(expf(top[threadIdx.x] - top[0]) / total));
    ids[threadIdx.x] = top_id[threadIdx.x];
  }
}

// out[h] (bfloat16) = sum over e of parts[e, h] * weights[e]: a decode row's experts mixed by their routing weights.
extern "C" __global__ void weighted_sum(const float* parts, const float* weights, int k, int H, unsigned short* out) {
  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  float acc = 0.0f;
  for (int e = 0; e < k; e++) acc += parts[(long long)e * H + h] * weights[e];
  out[h] = to_bf16(acc);
}

// The row's experts `ids` [k] of `layer` (of `layers`, `experts` each). Every expert keeps a use count that halves
// every `half_life` passes (score, as of step `at`, counted in layers: pass * layers + layer). A cached expert keeps its
// slot; a missed one takes the slot of the cached expert with the lowest count, or a free one, never one this layer used
// nor one being copied in. table holds a key's slot, -1, or -2 - slot while its slot is being copied in (pending).
// Without `cpu`, the GEMVs read a missed expert from host memory and fill its slot (plan code 1). With `cpu`, the most
// used of them do so too (their number set by `share`), running on the GPU while the CPU runs the others (code 2); of those, the ones
// used `admit` times or more (the most used first) reserve a slot for the host to copy them into, while budget[0] (the
// step's) lasts and at most budget[1] per layer, and plan[2e] is that slot (-1 without one). Otherwise plan[2e] is expert e's slot and code 0 means cached. The first
// layer ticks the pass clock and step gets this layer's step; reads counts hits, fills and CPU experts.
extern "C" __global__ void __launch_bounds__(1024) resolve_experts(const long long* ids, int k, int layer, int layers,
                                                                   int experts, int* table, int* owner, float* score,
                                                                   int* at, int* pending, int count, int* clock,
                                                                   int* budget, int* plan, int* reads, int* step,
                                                                   const float* share, double half_life, int cpu,
                                                                   double admit) {
  __shared__ int slot_of[32], fill[32], best_index[32];
  __shared__ float best_value[32], uses[32];
  __shared__ int now;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  int base = layer * experts;
  float log2_decay = -1.0f / ((float)half_life * layers);
  if (threadIdx.x == 0) {
    if (layer == 0) clock[0] += 1;
    now = clock[0] * layers + layer;
    step[0] = now;
  }
  __syncthreads();
  if (threadIdx.x < k) {
    int key = base + (int)ids[threadIdx.x];
    float kept = at[key] < 0 ? 0.0f : score[key] * exp2f(log2_decay * (float)(now - at[key]));
    score[key] = kept + 1.0f;
    uses[threadIdx.x] = kept + 1.0f;
    at[key] = now;
    int slot = table[key];
    slot_of[threadIdx.x] = slot >= 0 ? slot : -1;
    // 3: a CPU expert that may reserve a slot, 4: a missed expert not yet given to the GPU or the CPU
    fill[threadIdx.x] = slot >= 0 ? 0 : !cpu ? 1 : slot == -1 ? 4 : 2;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int ahead[32], misses = 0;
    for (int e = 0; e < k; e++) {
      ahead[e] = 0;
      for (int o = 0; o < k; o++)
        if (fill[e] == 4 && fill[o] == 4 && (uses[o] > uses[e] || (uses[o] == uses[e] && o < e))) ahead[e]++;
      misses += fill[e] == 4;
    }
    // the split that finishes first: share = cpu / (cpu + pcie) per expert, rounded towards the CPU by half a GPU expert
    int gpu = (int)(misses * share[0] + 0.5f * (1.0f - share[0]));
    for (int e = 0; e < k; e++)
      if (fill[e] == 4) fill[e] = ahead[e] < gpu ? 1 : uses[e] >= (float)admit ? 3 : 2;
    // the `allowed` most used of the CPU's keep their reservation
    int allowed = min(budget[0], budget[1]);
    for (int e = 0; e < k; e++) {
      if (fill[e] != 3) continue;
      int before = 0;
      for (int o = 0; o < k; o++)
        if (fill[o] == 3 && (uses[o] > uses[e] || (uses[o] == uses[e] && o < e))) before++;
      if (before < allowed) budget[0] -= 1;
      else fill[e] = 5;
    }
    for (int e = 0; e < k; e++) {
      if (fill[e] == 5) fill[e] = 2;
      atomicAdd(reads + (fill[e] == 0 ? 0 : fill[e] == 1 ? 1 : 2), 1);
    }
  }
  __syncthreads();
  for (int e = 0; e < k; e++) {
    if (fill[e] != 1 && fill[e] != 3) continue;
    float value = 3.0e38f;
    int index = INT_MAX;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
      int key = owner[i];
      float v = key < 0 ? -1.0f : pending[i] || at[key] == now ? 3.0e38f : score[key] * exp2f(log2_decay * (float)(now - at[key]));
      if (v < value) { value = v; index = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      float other_value = __shfl_xor_sync(0xffffffffu, value, o);
      int other_index = __shfl_xor_sync(0xffffffffu, index, o);
      if (other_value < value || (other_value == value && other_index < index)) { value = other_value; index = other_index; }
    }
    if (lane == 0) { best_value[warp] = value; best_index[warp] = index; }
    __syncthreads();
    if (warp == 0) {
      value = lane < warps ? best_value[lane] : 3.0e38f;
      index = lane < warps ? best_index[lane] : INT_MAX;
      for (int o = 16; o > 0; o >>= 1) {
        float other_value = __shfl_xor_sync(0xffffffffu, value, o);
        int other_index = __shfl_xor_sync(0xffffffffu, index, o);
        if (other_value < value || (other_value == value && other_index < index)) { value = other_value; index = other_index; }
      }
      // an expert goes without a slot when every slot is this layer's or pending (a small cache with the CPU side's
      // copies in flight): the GEMVs then read it from host memory without filling one
      if (lane == 0 && value < 3.0e38f) {
        int key = base + (int)ids[e];
        int old = owner[index];
        if (old >= 0) table[old] = -1;
        owner[index] = key;
        if (fill[e] == 1) {
          table[key] = index;
        } else {
          table[key] = -2 - index;
          pending[index] = 1;
        }
        slot_of[e] = index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < k) {
    plan[2 * threadIdx.x] = slot_of[threadIdx.x];
    plan[2 * threadIdx.x + 1] = fill[threadIdx.x] == 3 ? 2 : fill[threadIdx.x];
  }
}

// A decode layer's wait for the experts it left to the CPU (plan code 2): one thread polls `flag` (host memory) until
// the host sets it to this layer's step, for at most timeout_us, then the block copies the host's result (bfloat16 in
// host memory) to out [H]. Without such experts it returns at once. When the host is late, out is zeros and took_over
// makes the GEMVs run those experts themselves, so a slow or busy host never stalls or breaks the step.
extern "C" __global__ void await_cpu(const int* plan, int k, volatile const int* flag, const int* step,
                                     volatile const unsigned short* result, int timeout_us, float* out,
                                     int* took_over, int H) {
  __shared__ int ready;
  if (threadIdx.x == 0) {
    int wanted = 0;
    for (int e = 0; e < k; e++) wanted |= plan[2 * e + 1] == 2;
    ready = 0;
    if (wanted) {
      int expected = step[0];
      unsigned long long start, now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
      while (true) {
        if (*flag == expected) { ready = 1; break; }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if ((long long)(now - start) > 1000LL * timeout_us) break;
      }
      __threadfence_system();
    }
    took_over[0] = wanted && !ready;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < H; i += blockDim.x) out[i] = ready ? __int_as_float(((unsigned int)result[i]) << 16) : 0.0f;
}

// One block per row of x [rows, K] (float, or bfloat16 with `bf16`; with `swiglu`, float [rows, 2K] gate | up, SiLU(gate)
// * up first): the ConvRot rotation (the regular Hadamard of 256, h4 kron h4 kron h4 kron h4 / 16, per group) into out [rows, K]. The quantized
// experts are stored in that basis; their input stays in float (rounding it to int8 as well cost the 4-bit files a
// third of their main-token KL against FP8).
extern "C" __global__ void __launch_bounds__(256) rotate_rows(const void* x, int bf16, int swiglu, int K, float* out,
                                                              const int* only_if) {
  __shared__ float row[2048];
  int r = blockIdx.x;
  if (!only_if[0]) return;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    float v;
    if (swiglu) {
      const float* f = (const float*)x;
      float g = f[(long long)r * 2 * K + i], u = f[(long long)r * 2 * K + K + i];
      v = g / (1.0f + __expf(-g)) * u;
    } else if (bf16) {
      v = from_bf16(((const unsigned short*)x)[(long long)r * K + i]);
    } else {
      v = ((const float*)x)[(long long)r * K + i];
    }
    row[i] = v;
  }
  __syncthreads();
  for (int stride = 1; stride < 256; stride *= 4) {
    for (int t = threadIdx.x; t < K / 4; t += blockDim.x) {
      int within = t & 63, low = within % stride, high = within / stride;
      int i = (t >> 6) * 256 + high * stride * 4 + low;
      float a = row[i], b = row[i + stride], c = row[i + 2 * stride], d = row[i + 3 * stride];
      row[i] = a + b + c - d;
      row[i + stride] = a + b - c + d;
      row[i + 2 * stride] = a - b + c + d;
      row[i + 3 * stride] = -a + b + c + d;
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < K; i += blockDim.x) out[(long long)r * K + i] = row[i] * 0.0625f;
}

// The 16 int8 weights of an int4 against 16 floats of a row.
__device__ __forceinline__ float dot16(int4 q, const float4* x) {
  int words[4] = {q.x, q.y, q.z, q.w};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float4 xv = x[j];
    int word = words[j];
    acc += (float)(signed char)word * xv.x + (float)(signed char)(word >> 8) * xv.y +
           (float)(signed char)(word >> 16) * xv.z + (float)(word >> 24) * xv.w;
  }
  return acc;
}

// out[e, n] = expert e's int8 row n . x[e or 0] * row scale, x in float and rotated. Experts with plan fill are read
// from host memory and copied into their slot; those left to the CPU give zeros. With `late`, only those run (read from
// host memory) and only if the host was late (took_over); everything else gives zeros. Grid (N / ROWS_PER_BLOCK, k),
// block 256, shared memory 4 K bytes.
extern "C" __global__ void __launch_bounds__(256) gemv_int8(const float* x, int x_rows, const long long* ids,
                                                            const int* plan, const int* took_over, int late,
                                                            signed char* slot_w, float* slot_s,
                                                            const signed char* host_w, const float* host_s, float* out,
                                                            int N, int K) {
  extern __shared__ float4 xf[];
  int e = blockIdx.y, xr = x_rows == 1 ? 0 : e, code = plan[2 * e + 1];
  if (late ? code != 2 || !took_over[0] : code == 2) {
    if (threadIdx.x < ROWS_PER_BLOCK) out[(long long)e * N + blockIdx.x * ROWS_PER_BLOCK + threadIdx.x] = 0.0f;
    return;
  }
  const float4* x_src = (const float4*)(x + (long long)xr * K);
  for (int i = threadIdx.x; i < K / 4; i += blockDim.x) xf[i] = x_src[i];
  __syncthreads();
  int slot = plan[2 * e], fill = code == 1 && slot >= 0;
  long long id = ids[e];
  const signed char* w = code ? host_w + id * N * K : slot_w + (long long)slot * N * K;
  const float* s = code ? host_s + id * N : slot_s + (long long)slot * N;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int r = 0; r < ROWS_PER_WARP; r++) {
    int n = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP + r;
    const int4* w_row = (const int4*)(w + (long long)n * K);
    int4* dst = (int4*)(slot_w + ((long long)slot * N + n) * K);
    float acc = 0.0f;
    for (int v = lane; v < K / 16; v += 32) {
      int4 q = w_row[v];
      if (fill) dst[v] = q;
      acc += dot16(q, xf + 4 * v);
    }
    acc = warp_sum(acc);
    if (lane == 0) {
      float row_scale = s[n];
      out[(long long)e * N + n] = acc * row_scale;
      if (fill) slot_s[(long long)slot * N + n] = row_scale;
    }
  }
}

// out[e, n] = expert e's float8 row n . x[e or 0] with its 128 x 128 block scales [N / 128, K / 128]. Experts with plan
// fill are read from host memory and copied into their slot. Grid (N / ROWS_PER_BLOCK, k), block 256, shared memory
// 4 K bytes.
extern "C" __global__ void __launch_bounds__(256) gemv_fp8(const float* x, int x_rows, const long long* ids,
                                                           const int* plan, unsigned char* slot_w, float* slot_s,
                                                           const unsigned char* host_w, const float* host_s,
                                                           float* out, int N, int K) {
  extern __shared__ float4 xf[];
  __shared__ float block_scale[32];
  int e = blockIdx.y, xr = x_rows == 1 ? 0 : e;
  const float4* x_src = (const float4*)(x + (long long)xr * K);
  for (int i = threadIdx.x; i < K / 4; i += blockDim.x) xf[i] = x_src[i];
  int slot = plan[2 * e], host = plan[2 * e + 1], fill = host && slot >= 0;
  long long id = ids[e];
  int scale_cols = K / 128, scale_rows = N / 128;
  int first = blockIdx.x * ROWS_PER_BLOCK;
  const unsigned char* w = host ? host_w + id * N * K : slot_w + (long long)slot * N * K;
  const float* s = (host ? host_s + id * scale_rows * scale_cols : slot_s + (long long)slot * scale_rows * scale_cols)
                   + (first / 128) * scale_cols;
  if (threadIdx.x < scale_cols) {
    block_scale[threadIdx.x] = s[threadIdx.x];
    if (fill && first % 128 == 0)
      slot_s[((long long)slot * scale_rows + first / 128) * scale_cols + threadIdx.x] = s[threadIdx.x];
  }
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int r = 0; r < ROWS_PER_WARP; r++) {
    int n = first + warp * ROWS_PER_WARP + r;
    const uint4* w_row = (const uint4*)(w + (long long)n * K);
    uint4* dst = (uint4*)(slot_w + ((long long)slot * N + n) * K);
    float acc = 0.0f;
    for (int v = lane; v < K / 16; v += 32) {
      uint4 q = w_row[v];
      if (fill) dst[v] = q;
      unsigned int words[4] = {q.x, q.y, q.z, q.w};
      float part = 0.0f;
      for (int j = 0; j < 4; j++) {
        float4 xv = xf[4 * v + j];
        unsigned int word = words[j];
        part += e4m3(word & 0xffu) * xv.x + e4m3((word >> 8) & 0xffu) * xv.y + e4m3((word >> 16) & 0xffu) * xv.z +
                e4m3(word >> 24) * xv.w;
      }
      acc += part * block_scale[v / 8];
    }
    acc = warp_sum(acc);
    if (lane == 0) out[(long long)e * N + n] = acc;
  }
}

// Many rows, grouped by expert: group g is (expert, slot, fill, first row, rows) over x (float, rotated), whose rows
// are sorted by expert. out[r, n] = row n of the group's expert . x[r] * row scale. A group reads its expert from its
// slot, or from host memory when it has no slot (-1) or fills it, copying it in. Grid (N / ROWS_PER_BLOCK, groups),
// block 256; K <= 2048.
extern "C" __global__ void __launch_bounds__(256) grouped_int8(const float* x, const int* groups, signed char* slot_w,
                                                               float* slot_s, const signed char* host_w,
                                                               const float* host_s, float* out, int N, int K) {
  const int* group = groups + 5 * blockIdx.y;
  int expert = group[0], slot = group[1], fill = group[2], first = group[3], rows = group[4];
  bool host = slot < 0 || fill;
  const signed char* w = host ? host_w + (long long)expert * N * K : slot_w + (long long)slot * N * K;
  const float* s = host ? host_s + (long long)expert * N : slot_s + (long long)slot * N;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, vectors = K / 16;
  for (int r = 0; r < ROWS_PER_WARP; r++) {
    int n = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP + r;
    const int4* w_row = (const int4*)(w + (long long)n * K);
    int4* dst = (int4*)(slot_w + ((long long)slot * N + n) * K);
    int4 wv[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int v = lane + 32 * i;
      wv[i] = make_int4(0, 0, 0, 0);
      if (v < vectors) {
        wv[i] = w_row[v];
        if (fill) dst[v] = wv[i];
      }
    }
    float row_scale = s[n];
    if (fill && lane == 0) slot_s[(long long)slot * N + n] = row_scale;
    for (int t = first; t < first + rows; t++) {
      const float4* x_row = (const float4*)(x + (long long)t * K);
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int v = lane + 32 * i;
        if (v < vectors) acc += dot16(wv[i], x_row + 4 * v);
      }
      acc = warp_sum(acc);
      if (lane == 0) out[(long long)t * N + n] = acc * row_scale;
    }
  }
}

// grouped_int8 for float8 experts with 128 x 128 block scales, against float rows x.
extern "C" __global__ void __launch_bounds__(256) grouped_fp8(const float* x, const int* groups, unsigned char* slot_w,
                                                              float* slot_s, const unsigned char* host_w,
                                                              const float* host_s, float* out, int N, int K) {
  __shared__ float block_scale[32];
  const int* group = groups + 5 * blockIdx.y;
  int expert = group[0], slot = group[1], fill = group[2], first = group[3], rows = group[4];
  bool host = slot < 0 || fill;
  int scale_cols = K / 128, scale_rows = N / 128, first_row = blockIdx.x * ROWS_PER_BLOCK;
  const unsigned char* w = host ? host_w + (long long)expert * N * K : slot_w + (long long)slot * N * K;
  const float* s = (host ? host_s + (long long)expert * scale_rows * scale_cols
                         : slot_s + (long long)slot * scale_rows * scale_cols) + (first_row / 128) * scale_cols;
  if (threadIdx.x < scale_cols) {
    block_scale[threadIdx.x] = s[threadIdx.x];
    if (fill && first_row % 128 == 0)
      slot_s[((long long)slot * scale_rows + first_row / 128) * scale_cols + threadIdx.x] = s[threadIdx.x];
  }
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, vectors = K / 16;
  for (int r = 0; r < ROWS_PER_WARP; r++) {
    int n = first_row + warp * ROWS_PER_WARP + r;
    const uint4* w_row = (const uint4*)(w + (long long)n * K);
    uint4* dst = (uint4*)(slot_w + ((long long)slot * N + n) * K);
    float wf[4][16];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int v = lane + 32 * i;
      uint4 q = make_uint4(0, 0, 0, 0);
      float sc = 0.0f;
      if (v < vectors) {
        q = w_row[v];
        if (fill) dst[v] = q;
        sc = block_scale[v / 8];
      }
      unsigned int words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int j = 0; j < 16; j++) wf[i][j] = e4m3((words[j / 4] >> (8 * (j % 4))) & 0xffu) * sc;
    }
    for (int t = first; t < first + rows; t++) {
      const float4* x_row = (const float4*)(x + (long long)t * K);
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int v = lane + 32 * i;
        if (v < vectors) {
#pragma unroll
          for (int j = 0; j < 4; j++) {
            float4 xv = x_row[4 * v + j];
            acc += wf[i][4 * j] * xv.x + wf[i][4 * j + 1] * xv.y + wf[i][4 * j + 2] * xv.z + wf[i][4 * j + 3] * xv.w;
          }
        }
      }
      acc = warp_sum(acc);
      if (lane == 0) out[(long long)t * N + n] = acc;
    }
  }
}

// 16 columns (vector v) of a packed W4A8/W6A8 row on the int8 grid, as comfy_kitchen decodes them: the code (a nibble,
// with 6 bits plus two bits from the high plane after the K/2 nibble bytes) gives a value (4 bits: the bank's codebook;
// 6 bits: code - 32), times the group's e4m3 scale, rounded and clamped to +-127.
__device__ __forceinline__ int4 decode_codes(const unsigned char* row, unsigned int group_scale, int v, int K, int bits,
                                             const float* codebook) {
  uint2 low = *(const uint2*)(row + v * 8);
  unsigned int high = bits == 6 ? *(const unsigned int*)(row + K / 2 + v * 4) : 0u;
  float s = e4m3(group_scale);
  unsigned int packed[4] = {0u, 0u, 0u, 0u};
#pragma unroll
  for (int j = 0; j < 16; j++) {
    unsigned int code = ((j < 8 ? low.x : low.y) >> (4 * (j & 7))) & 0xFu;
    float value = bits == 6 ? (float)((int)(code | (((high >> (2 * j)) & 3u) << 4)) - 32) : codebook[code];
    int q = max(-127, min(127, __float2int_rn(value * s)));
    packed[j >> 2] |= ((unsigned int)q & 0xFFu) << (8 * (j & 3));
  }
  return make_int4((int)packed[0], (int)packed[1], (int)packed[2], (int)packed[3]);
}

// The warp's copy of a packed row (row_bytes), its group scales (K / 16 bytes) into a slot; both sizes are multiples of
// 16 bytes.
__device__ __forceinline__ void copy_codes_row(const unsigned char* w, const unsigned char* r, unsigned char* dw,
                                               unsigned char* dr, int row_bytes, int groups, int lane) {
  for (int i = lane; i < row_bytes / 16; i += 32) ((int4*)dw)[i] = ((const int4*)w)[i];
  for (int i = lane; i < groups / 16; i += 32) ((int4*)dr)[i] = ((const int4*)r)[i];
  __threadfence_block();
  __syncwarp();
}

// gemv_int8 for W4A8/W6A8 banks (fields: packed rows, e4m3 group scales, per-row channel scales; a 16-entry codebook
// per bank at 4 bits). A filled expert is copied into its slot first, then read from there.
extern "C" __global__ void __launch_bounds__(256) gemv_codes(const float* x, int x_rows, const long long* ids,
                                                             const int* plan, unsigned char* slot_w,
                                                             unsigned char* slot_r, float* slot_c,
                                                             const unsigned char* host_w, const unsigned char* host_r,
                                                             const float* host_c, const float* codebook, int bits,
                                                             float* out, int N, int K) {
  extern __shared__ float4 xf[];
  __shared__ float lut[16];
  int e = blockIdx.y, xr = x_rows == 1 ? 0 : e;
  const float4* x_src = (const float4*)(x + (long long)xr * K);
  for (int i = threadIdx.x; i < K / 4; i += blockDim.x) xf[i] = x_src[i];
  if (bits == 4 && threadIdx.x < 16) lut[threadIdx.x] = codebook[threadIdx.x];
  __syncthreads();
  int slot = plan[2 * e], host = plan[2 * e + 1], fill = host && slot >= 0, row_bytes = K * bits / 8, groups = K / 16;
  long long id = ids[e];
  const unsigned char* w = host ? host_w + id * N * row_bytes : slot_w + (long long)slot * N * row_bytes;
  const unsigned char* r = host ? host_r + id * N * groups : slot_r + (long long)slot * N * groups;
  const float* c = host ? host_c + id * N : slot_c + (long long)slot * N;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int k = 0; k < ROWS_PER_WARP; k++) {
    int n = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP + k;
    const unsigned char* w_row = w + (long long)n * row_bytes;
    const unsigned char* r_row = r + (long long)n * groups;
    if (fill) {
      unsigned char* dw = slot_w + ((long long)slot * N + n) * row_bytes;
      unsigned char* dr = slot_r + ((long long)slot * N + n) * groups;
      copy_codes_row(w_row, r_row, dw, dr, row_bytes, groups, lane);
      w_row = dw;
      r_row = dr;
    }
    float acc = 0.0f;
    for (int v = lane; v < K / 16; v += 32) acc += dot16(decode_codes(w_row, r_row[v], v, K, bits, lut), xf + 4 * v);
    acc = warp_sum(acc);
    if (lane == 0) {
      float channel = c[n];
      out[(long long)e * N + n] = acc * channel;
      if (fill) slot_c[(long long)slot * N + n] = channel;
    }
  }
}

// grouped_int8 for W4A8/W6A8 banks (see gemv_codes); K <= 2048.
extern "C" __global__ void __launch_bounds__(256) grouped_codes(const float* x, const int* groups,
                                                                unsigned char* slot_w, unsigned char* slot_r,
                                                                float* slot_c, const unsigned char* host_w,
                                                                const unsigned char* host_r, const float* host_c,
                                                                const float* codebook, int bits, float* out, int N,
                                                                int K) {
  __shared__ float lut[16];
  if (bits == 4 && threadIdx.x < 16) lut[threadIdx.x] = codebook[threadIdx.x];
  __syncthreads();
  const int* group = groups + 5 * blockIdx.y;
  int expert = group[0], slot = group[1], fill = group[2], first = group[3], rows = group[4];
  int row_bytes = K * bits / 8, scale_groups = K / 16, vectors = K / 16;
  bool host = slot < 0 || fill;
  const unsigned char* w = host ? host_w + (long long)expert * N * row_bytes : slot_w + (long long)slot * N * row_bytes;
  const unsigned char* r = host ? host_r + (long long)expert * N * scale_groups : slot_r + (long long)slot * N * scale_groups;
  const float* c = host ? host_c + (long long)expert * N : slot_c + (long long)slot * N;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int k = 0; k < ROWS_PER_WARP; k++) {
    int n = blockIdx.x * ROWS_PER_BLOCK + warp * ROWS_PER_WARP + k;
    const unsigned char* w_row = w + (long long)n * row_bytes;
    const unsigned char* r_row = r + (long long)n * scale_groups;
    if (fill) {
      unsigned char* dw = slot_w + ((long long)slot * N + n) * row_bytes;
      unsigned char* dr = slot_r + ((long long)slot * N + n) * scale_groups;
      copy_codes_row(w_row, r_row, dw, dr, row_bytes, scale_groups, lane);
      w_row = dw;
      r_row = dr;
    }
    int4 wv[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int v = lane + 32 * i;
      wv[i] = v < vectors ? decode_codes(w_row, r_row[v], v, K, bits, lut) : make_int4(0, 0, 0, 0);
    }
    float channel = c[n];
    if (fill && lane == 0) slot_c[(long long)slot * N + n] = channel;
    for (int t = first; t < first + rows; t++) {
      const float4* x_row = (const float4*)(x + (long long)t * K);
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int v = lane + 32 * i;
        if (v < vectors) acc += dot16(wv[i], x_row + 4 * v);
      }
      acc = warp_sum(acc);
      if (lane == 0) out[(long long)t * N + n] = acc * channel;
    }
  }
}
"""

ROWS_PER_BLOCK = 16


@functools.cache
def _kernel(name, device_index):
    with torch.cuda.device(device_index):
        return torch.cuda._compile_kernel(SOURCE, name)


def _launch(name, device, grid, block, args, shared_mem=0):
    _kernel(name, device.index)(grid=grid, block=block, args=args, shared_mem=shared_mem,
                                stream=torch.cuda.current_stream(device))


@functools.cache
def available(device):
    """Whether the kernels compile for `device`: a CUDA device and NVRTC."""
    if device.type != "cuda":
        return False
    try:
        for name in ("route_experts", "weighted_sum", "resolve_experts", "rotate_rows", "gemv_int8", "gemv_fp8", "grouped_int8", "grouped_fp8", "await_cpu", "gemv_codes",
                     "grouped_codes"):
            _kernel(name, device.index)
    except (AttributeError, RuntimeError, OSError):  # PyTorch without the NVRTC helper, or NVRTC missing
        return False
    return True


def resolve(ids, layer, layers, num_experts, half_life, cpu, admit, table, owner, score, at, pending, clock, budget, reads, step,
            share):
    """plan [k, 2] int32 for the row's experts `ids` [k] (int64) of `layer`; see resolve_experts."""
    plan = torch.empty((ids.shape[0], 2), dtype=torch.int32, device=ids.device)
    _launch("resolve_experts", ids.device, (1, 1, 1), (1024, 1, 1),
            [ids, ids.shape[0], layer, layers, num_experts, table, owner, score, at, pending, owner.shape[0], clock, budget,
             plan, reads, step, share, float(half_life), int(cpu), float(admit)])
    return plan


def await_cpu(plan, flag, step, result, timeout_us, took_over):
    """[1, H] float32: the host's result for the experts `plan` left to the CPU, or zeros (see await_cpu)."""
    out = torch.empty((1, result.shape[-1]), dtype=torch.float32, device=plan.device)
    _launch("await_cpu", plan.device, (1, 1, 1), (256, 1, 1),
            [plan, plan.shape[0], flag, step, result, timeout_us, out, took_over, result.shape[-1]])
    return out


def rotate(x, only_if, swiglu=False):
    """Rows of x [rows, K] (float32 or bfloat16; float32 [rows, 2K] gate | up with `swiglu`, SiLU(gate) * up first) in the
    ConvRot basis: float32 [rows, K]. Only when `only_if` (an int32 on the GPU) is set; otherwise the output is left
    unset."""
    rows, width = x.shape
    k = width // 2 if swiglu else width
    out = torch.empty((rows, k), dtype=torch.float32, device=x.device)
    _launch("rotate_rows", x.device, (rows, 1, 1), (256, 1, 1), [x, int(x.dtype == torch.bfloat16), int(swiglu), k, out, only_if])
    return out


def route(logits, k):
    """(ids [1, k] int64, weights [1, k] float32) of one row of router logits [1, n] (bfloat16): the k largest, by
    descending logit, and their softmax weights renormalized over them, rounded to bfloat16."""
    ids = torch.empty((1, k), dtype=torch.int64, device=logits.device)
    weights = torch.empty((1, k), dtype=torch.float32, device=logits.device)
    _launch("route_experts", logits.device, (1, 1, 1), (256, 1, 1), [logits, logits.shape[1], k, ids, weights])
    return ids, weights


def weighted_sum(parts, weights):
    """[1, H] bfloat16: parts [k, H] (float32) summed by weights [k] (float32)."""
    out = torch.empty((1, parts.shape[1]), dtype=torch.bfloat16, device=parts.device)
    _launch("weighted_sum", parts.device, ((parts.shape[1] + 255) // 256, 1, 1), (256, 1, 1),
            [parts, weights, parts.shape[0], parts.shape[1], out])
    return out


def gemv_int8(x, ids, plan, took_over, late, slot_fields, host_fields, n):
    """[k, n] float32: each expert's int8 rows against x [1 or k, K] (float32, rotated); fields are {"_qdata", "scale"}."""
    k, width = ids.shape[0], x.shape[1]
    out = torch.empty((k, n), dtype=torch.float32, device=x.device)
    _launch("gemv_int8", x.device, (n // ROWS_PER_BLOCK, k, 1), (256, 1, 1),
            [x, x.shape[0], ids, plan, took_over, int(late), slot_fields["_qdata"], slot_fields["scale"],
             host_fields["_qdata"], host_fields["scale"], out, n, width], shared_mem=4 * width)
    return out


def gemv_fp8(x, ids, plan, slot_fields, host_fields, n):
    """[k, n] float32: each expert's float8 rows against x [1 or k, K] float32; fields are {"weight", "scale"}."""
    k, width = ids.shape[0], x.shape[1]
    out = torch.empty((k, n), dtype=torch.float32, device=x.device)
    _launch("gemv_fp8", x.device, (n // ROWS_PER_BLOCK, k, 1), (256, 1, 1),
            [x, x.shape[0], ids, plan, slot_fields["weight"], slot_fields["scale"], host_fields["weight"],
             host_fields["scale"], out, n, width], shared_mem=4 * width)
    return out


def grouped_int8(x, groups, slot_fields, host_fields, n):
    """[rows, n] float32 for x's rows (float32, rotated, sorted by expert) through their experts; groups [g, 5] int32 as
    grouped_int8. Rows outside every group are left unset."""
    out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
    _launch("grouped_int8", x.device, (n // ROWS_PER_BLOCK, groups.shape[0], 1), (256, 1, 1),
            [x, groups, slot_fields["_qdata"], slot_fields["scale"], host_fields["_qdata"], host_fields["scale"], out, n,
             x.shape[1]])
    return out


def grouped_fp8(x, groups, slot_fields, host_fields, n):
    """grouped_int8 for float8 experts, against float32 rows x."""
    out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
    _launch("grouped_fp8", x.device, (n // ROWS_PER_BLOCK, groups.shape[0], 1), (256, 1, 1),
            [x, groups, slot_fields["weight"], slot_fields["scale"], host_fields["weight"], host_fields["scale"], out, n,
             x.shape[1]])
    return out


def gemv_codes(x, ids, plan, slot_fields, host_fields, codebook, bits, n):
    """gemv_int8 for W4A8/W6A8 banks; fields are {"_qdata", "scale" (e4m3), "s_channel"}, codebook [16] at 4 bits."""
    k, width = ids.shape[0], x.shape[1]
    if codebook is None:
        codebook = host_fields["s_channel"]  # never read at 6 bits; the launcher needs a pointer
    out = torch.empty((k, n), dtype=torch.float32, device=x.device)
    _launch("gemv_codes", x.device, (n // ROWS_PER_BLOCK, k, 1), (256, 1, 1),
            [x, x.shape[0], ids, plan, slot_fields["_qdata"], slot_fields["scale"], slot_fields["s_channel"],
             host_fields["_qdata"], host_fields["scale"], host_fields["s_channel"], codebook, bits, out, n, width],
            shared_mem=4 * width)
    return out


def grouped_codes(x, groups, slot_fields, host_fields, codebook, bits, n):
    """grouped_int8 for W4A8/W6A8 banks (see gemv_codes)."""
    out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
    if codebook is None:
        codebook = host_fields["s_channel"]  # never read at 6 bits
    _launch("grouped_codes", x.device, (n // ROWS_PER_BLOCK, groups.shape[0], 1), (256, 1, 1),
            [x, groups, slot_fields["_qdata"], slot_fields["scale"], slot_fields["s_channel"], host_fields["_qdata"],
             host_fields["scale"], host_fields["s_channel"], codebook, bits, out, n, x.shape[1]])
    return out
