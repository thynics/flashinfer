/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// A long row never enters the legacy 18-bit, epoch-tagged selector. P4
// publishes complete scores/summaries through C4 before this scratch is read.
namespace flashinfer::glm5_tp8::indexer_q4::long_topk {
constexpr int kTopK = 2048;
constexpr int kGroup = 32;
constexpr int kPool = (1048576 + 4096) / kGroup / 2 + 8;
constexpr int kThreads = 384;
constexpr int kWarps = kThreads / 32;
constexpr int kLogicalMax = 1048576 + 4096;

__host__ __device__ constexpr int group_count(int length) { return (length + kGroup - 1) / kGroup; }
// A row holds the group maxima at uint32 [0, G) and one auxiliary word per
// causal group from the 16-byte aligned uint32 offset (G_causal + 4) & ~3,
// which lies past every maximum because G <= G_causal + 1. The stride keeps
// both regions in the row: 2 * stride >= 2 * G + 8.
__host__ __device__ constexpr int summary_stride(int length) {
  return (group_count(length) + 5) & ~1;
}
__device__ __forceinline__ uint32_t score_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
__device__ __forceinline__ uint64_t composite(float value, uint32_t id) {
  // Preserve signed-zero ordering and choose smaller IDs on score ties.
  return (uint64_t(score_key(value)) << 32) | (0xffffffffu - id);
}
__device__ __forceinline__ uint64_t load_cg(const uint64_t* p) {
  uint64_t x;
  asm volatile("ld.global.cg.b64 %0, [%1];" : "=l"(x) : "l"(p) : "memory");
  return x;
}
__device__ __forceinline__ float load_cg(const float* p) {
  float x;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x) : "l"(p) : "memory");
  return x;
}
__device__ __forceinline__ void store_cg(uint64_t* p, uint64_t x) {
  asm volatile("st.global.cg.b64 [%0], %1;" ::"l"(p), "l"(x) : "memory");
}
__device__ __forceinline__ uint64_t warp_down(uint64_t x, int delta, int width) {
  const uint32_t lo = __shfl_down_sync(0xffffffffu, uint32_t(x), delta, width);
  const uint32_t hi = __shfl_down_sync(0xffffffffu, uint32_t(x >> 32), delta, width);
  return uint64_t(lo) | (uint64_t(hi) << 32);
}

// Every lane participates, including the final partial 128-row tile. A group
// is one warp (lane 0 sits on a multiple of 32 rows) and has exactly one
// writer; release/acquire at C4 makes atomic publication tags unnecessary.
// The summary row holds one full 32-bit score key per group: for disjoint
// contiguous groups, (max key, smaller group index) orders groups exactly as
// their maximum composites do. The auxiliary word holds the lowest lane with
// the maximum key (bits 31..27) and the second-largest key of the group
// (largest key of every other lane) truncated to bits 31..5, so the true
// second key is below ((word & 0x07ffffff) + 1) << 5. Invalid rows have key 0,
// below every finite score key. Never preload these summaries before C4.
__device__ __forceinline__ void publish_group(float score, int position, int valid_rows,
                                              int physical_rows, uint64_t* summary) {
  const uint32_t key = position < valid_rows ? score_key(score) : 0u;
  uint32_t top = key;
#pragma unroll
  for (int offset = kGroup / 2; offset > 0; offset >>= 1)
    top = max(top, __shfl_xor_sync(0xffffffffu, top, offset));
  const uint32_t lane = threadIdx.x & (kGroup - 1);
  const uint32_t arg = uint32_t(__ffs(__ballot_sync(0xffffffffu, key == top)) - 1);
  uint32_t second = lane == arg ? 0u : key;
#pragma unroll
  for (int offset = kGroup / 2; offset > 0; offset >>= 1)
    second = max(second, __shfl_xor_sync(0xffffffffu, second, offset));
  if (lane == 0u && position < valid_rows && position < physical_rows) {
    uint32_t* const row = reinterpret_cast<uint32_t*>(summary);
    const int group = position / kGroup;
    __stcg(row + group, top);
    __stcg(row + ((group_count(valid_rows) + 4) & ~3) + group, (arg << 27) | (second >> 5));
  }
}

struct alignas(16) Scratch {
  uint64_t candidates[kPool];
  uint32_t hist[2048];
  uint32_t groups[kTopK];
  uint32_t group_keys[kTopK];
  uint32_t multi_groups[kTopK];
  uint32_t warp_hist[kWarps][256];
  uint32_t partial[kWarps];
  uint32_t count;
  uint32_t remaining;
  uint32_t bucket_count;
  uint32_t multi;
  uint32_t digit;
  uint32_t key_max;
  uint32_t key_min;
  uint32_t key_floor;
  uint32_t pool;
  uint32_t reserved;
  uint64_t prefix;
  uint64_t mask;
  uint64_t cutoff;
};
static_assert(sizeof(Scratch) + 0x3c00 <= 0x38000, "long selector fits existing shared buffer");

struct ScoreRow {
  const float* data;
  __device__ __forceinline__ uint64_t operator()(int i) const {
    return composite(load_cg(data + i), uint32_t(i));
  }
};
struct SummaryRow {
  const uint64_t* data;
  __device__ __forceinline__ uint64_t operator()(int i) const { return load_cg(data + i); }
};
struct CandidateRow {
  const uint64_t* data;
  __device__ __forceinline__ uint64_t operator()(int i) const { return data[i]; }
};

// Exact descending radix rank. Histogram population is aggregated within a
// warp before a shared atomic, so constant score bytes do not serialize 32
// identical updates. One CTA owns a query; no inter-CTA barrier is added.
// One warp scans the 256 bins in parallel. Once the target bucket holds
// exactly the remaining count, every key in it is selected, so the prefix
// with a zero suffix is already the exact cutoff for emit (value >= cutoff).
// Callers store whole leading bytes shared by every key in s.prefix/s.mask
// (zero when unknown) and synchronize before entry; those bytes need no pass.
template <class Row>
__device__ __forceinline__ uint64_t kth(Row row, int count, Scratch& s) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  uint64_t prefix = s.prefix, mask = s.mask;
  uint32_t remaining = kTopK;
  for (int shift = 56; shift >= 0; shift -= 8) {
    if (((mask >> shift) & 255u) == 255u) continue;
    for (int i = tid; i < kWarps * 256; i += kThreads)
      reinterpret_cast<uint32_t*>(s.warp_hist)[i] = 0;
    __syncthreads();
    for (int base = warp * 32; base < count; base += kThreads) {
      const int i = base + lane;
      const uint64_t value = i < count ? row(i) : 0;
      const bool match = i < count && (value & mask) == prefix;
      const unsigned active = __ballot_sync(0xffffffffu, match);
      if (match) {
        const unsigned digit = unsigned(value >> shift) & 255u;
        const unsigned peers = __match_any_sync(active, digit);
        if (lane == __ffs(peers) - 1) atomicAdd(&s.warp_hist[warp][digit], unsigned(__popc(peers)));
      }
    }
    __syncthreads();
    if (tid < 256) {
      uint32_t sum = 0;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) sum += s.warp_hist[w][tid];
      s.hist[tid] = sum;
    }
    __syncthreads();
    if (warp == 0) {
      uint32_t local = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) local += s.hist[255 - 8 * lane - j];
      uint32_t incl = local;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const uint32_t other = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset) incl += other;
      }
      const bool hit = incl - local < remaining && incl >= remaining;
      if (hit) {
        uint32_t left = remaining - (incl - local);
        for (int j = 0; j < 8; ++j) {
          const int digit = 255 - 8 * lane - j;
          const uint32_t n = s.hist[digit];
          if (left > n) {
            left -= n;
          } else {
            s.prefix = prefix | (uint64_t(digit) << shift);
            s.remaining = left;
            s.bucket_count = n;
            break;
          }
        }
      }
      if (__ballot_sync(0xffffffffu, hit) == 0u) __trap();
    }
    __syncthreads();
    prefix = s.prefix;
    remaining = s.remaining;
    mask |= uint64_t(255) << shift;
    if (s.bucket_count == remaining) return prefix;
  }
  return prefix;
}

template <class Row>
__device__ __forceinline__ void emit(Row row, int count, uint64_t cutoff, int32_t* out,
                                     Scratch& s) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  if (tid == 0) s.count = 0;
  __syncthreads();
  for (int base = warp * 32; base < count; base += kThreads) {
    const int i = base + lane;
    const uint64_t value = i < count ? row(i) : 0;
    const bool take = i < count && value >= cutoff;
    const unsigned ballot = __ballot_sync(0xffffffffu, take);
    if (ballot != 0u) {
      uint32_t slot = 0;
      if (lane == 0) slot = atomicAdd(&s.count, unsigned(__popc(ballot)));
      slot = __shfl_sync(0xffffffffu, slot, 0) + unsigned(__popc(ballot & ((1u << lane) - 1u)));
      if (take && slot < kTopK) out[slot] = int32_t(0xffffffffu - uint32_t(value));
    }
  }
  __syncthreads();
  if (s.count != kTopK) __trap();
}

// Keep the large radix body outside the persistent kernel's short-row path.
// Grouped path: stage all 32-row group keys in shared memory once, rank them
// exactly (adaptive 11-bit digits, smaller group index on equal keys), then
// build the pool of items whose key reaches the smallest selected group key.
// A selected group whose second key is below that floor contributes exactly
// its maximum, taken from the summary; only the other selected groups are
// gathered. The bounded pool is then ranked. A pool that overflows the bounded
// storage (concentrated ties) falls back to the exact full-row rank.
static __device__ __noinline__ void select(const float* scores, int rows, const uint64_t* maxima,
                                           int32_t* out, void* scratch, bool grouped) {
  Scratch& s = *reinterpret_cast<Scratch*>(scratch);
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  if (rows <= kTopK) {
    for (int i = tid; i < kTopK; i += kThreads) out[i] = i < rows ? i : -1;
    __syncthreads();
    return;
  }
  if (!grouped || group_count(rows) < kTopK) {
    if (tid == 0) {
      s.prefix = 0;
      s.mask = 0;
    }
    __syncthreads();
    const uint64_t cutoff = kth(ScoreRow{scores}, rows, s);
    emit(ScoreRow{scores}, rows, cutoff, out, s);
    return;
  }
  const int ngroups = group_count(rows);
  uint32_t* const keys = reinterpret_cast<uint32_t*>(s.candidates);
  const uint32_t* const summary = reinterpret_cast<const uint32_t*>(maxima);
  if (tid == 0) {
    s.key_max = 0u;
    s.key_min = 0xffffffffu;
    s.key_floor = 0xffffffffu;
    s.count = 0u;
    s.pool = 0u;
    s.multi = 0u;
  }
  uint32_t kmax = 0u, kmin = 0xffffffffu;
  for (int base = tid; base < ngroups; base += kThreads * 32) {
    uint32_t v[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int i = base + j * kThreads;
      v[j] = i < ngroups ? __ldcg(summary + i) : 0u;
    }
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      const int i = base + j * kThreads;
      if (i < ngroups) {
        keys[i] = v[j];
        kmax = max(kmax, v[j]);
        kmin = min(kmin, v[j]);
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    kmax = max(kmax, __shfl_xor_sync(0xffffffffu, kmax, offset));
    kmin = min(kmin, __shfl_xor_sync(0xffffffffu, kmin, offset));
  }
  __syncthreads();
  if (lane == 0) {
    atomicMax(&s.key_max, kmax);
    atomicMin(&s.key_min, kmin);
  }
  __syncthreads();
  const uint32_t key_max = s.key_max, key_min = s.key_min;

  // Bits above the highest bit where keys differ are common to every key.
  uint32_t prefix = key_max, mask = 0xffffffffu, remaining = kTopK, bucket = uint32_t(ngroups);
  int shift = 0;
  if (key_max != key_min) {
    shift = 32 - __clz(int(key_max ^ key_min));
    mask = uint32_t(0xffffffffull << shift);
    prefix = key_max & mask;
  }
  while (shift > 0 && bucket != remaining) {
    const int width = shift < 11 ? shift : 11;
    shift -= width;
    for (int i = tid; i < 2048; i += kThreads) s.hist[i] = 0u;
    __syncthreads();
    for (int base = tid; base < ngroups; base += kThreads * 4) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = base + j * kThreads;
        if (i < ngroups) {
          const uint32_t key = keys[i];
          if ((key & mask) == prefix) atomicAdd(&s.hist[(key >> shift) & ((1u << width) - 1u)], 1u);
        }
      }
    }
    __syncthreads();
    // Parallel descending scan: thread t < 256 owns digits 2047 - 8t - j, j < 8,
    // read as two conflict-free 16-byte words.
    uint32_t h[8];
    if (tid < 256) {
      const uint4 a = *reinterpret_cast<const uint4*>(s.hist + (2044 - 8 * tid));
      const uint4 b = *reinterpret_cast<const uint4*>(s.hist + (2040 - 8 * tid));
      h[0] = a.w;
      h[1] = a.z;
      h[2] = a.y;
      h[3] = a.x;
      h[4] = b.w;
      h[5] = b.z;
      h[6] = b.y;
      h[7] = b.x;
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) h[j] = 0u;
    }
    uint32_t local = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) local += h[j];
    uint32_t incl = local;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const uint32_t other = __shfl_up_sync(0xffffffffu, incl, offset);
      if (lane >= offset) incl += other;
    }
    if (lane == 31) s.partial[warp] = incl;
    __syncthreads();
    uint32_t before = incl - local;
    for (int w = 0; w < warp; ++w) before += s.partial[w];
    const bool hit = before < remaining && before + local >= remaining;
    if (hit) {
      uint32_t left = remaining - before;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (left > h[j]) {
          left -= h[j];
        } else {
          s.digit = uint32_t(2047 - 8 * tid - j);
          s.remaining = left;
          s.bucket_count = h[j];
          break;
        }
      }
    }
    if (__syncthreads_count(hit) != 1) __trap();
    prefix |= s.digit << shift;
    mask |= ((1u << width) - 1u) << shift;
    remaining = s.remaining;
    bucket = s.bucket_count;
  }
  // Unresolved only when all 32 key bits are fixed and the key is shared.
  const bool ties = bucket != remaining;

  uint32_t floor_key = 0xffffffffu;
  for (int base = warp * 32; base < ngroups; base += kThreads) {
    const int i = base + lane;
    const uint32_t key = i < ngroups ? keys[i] : 0u;
    const bool take = i < ngroups && (ties ? key > prefix : key >= prefix);
    const unsigned ballot = __ballot_sync(0xffffffffu, take);
    if (ballot != 0u) {
      uint32_t slot = 0u;
      if (lane == 0) slot = atomicAdd(&s.count, unsigned(__popc(ballot)));
      slot = __shfl_sync(0xffffffffu, slot, 0) + unsigned(__popc(ballot & ((1u << lane) - 1u)));
      if (take) {
        if (slot < kTopK) {
          s.groups[slot] = uint32_t(i);
          s.group_keys[slot] = key;
        }
        floor_key = min(floor_key, key);
      }
    }
  }
  if (ties) {
    // Equal keys: the lowest group indices win. Rank contiguous chunks.
    const int chunk = (ngroups + kThreads - 1) / kThreads;
    const int begin = min(tid * chunk, ngroups), end = min(begin + chunk, ngroups);
    uint32_t mine = 0u;
    for (int i = begin; i < end; ++i) mine += keys[i] == prefix ? 1u : 0u;
    uint32_t incl = mine;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const uint32_t other = __shfl_up_sync(0xffffffffu, incl, offset);
      if (lane >= offset) incl += other;
    }
    if (lane == 31) s.partial[warp] = incl;
    __syncthreads();
    uint32_t rank = incl - mine;
    for (int w = 0; w < warp; ++w) rank += s.partial[w];
    for (int i = begin; i < end && rank < remaining; ++i) {
      if (keys[i] == prefix) {
        const uint32_t slot = atomicAdd(&s.count, 1u);
        if (slot < kTopK) {
          s.groups[slot] = uint32_t(i);
          s.group_keys[slot] = prefix;
        }
        ++rank;
      }
    }
    floor_key = min(floor_key, prefix);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    floor_key = min(floor_key, __shfl_xor_sync(0xffffffffu, floor_key, offset));
  if (lane == 0) atomicMin(&s.key_floor, floor_key);
  __syncthreads();
  if (s.count != kTopK) __trap();
  floor_key = s.key_floor;

  // The 2048 selected maxima all have key >= floor_key, so the K-th item does
  // too, and every top-K item lies in a selected group. The staged keys are
  // dead from here on; the pool reuses their storage. A group whose second
  // key is below the floor adds only its maximum: the summary key and the ID
  // of its lowest maximal lane form that item's exact composite.
  {
    const uint32_t* const aux = summary + ((ngroups + 4) & ~3);
    uint32_t word[6];
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      const int i = (warp + j * kWarps) * 32 + lane;
      word[j] = i < kTopK ? __ldcg(aux + s.groups[i]) : 0u;
    }
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      const int i = (warp + j * kWarps) * 32 + lane;
      if (i < kTopK) {
        const uint32_t g = s.groups[i];
        const bool single = (uint64_t((word[j] & 0x07ffffffu) + 1u) << 5) <= uint64_t(floor_key);
        const unsigned ones = __ballot_sync(0xffffffffu, single);
        if (ones != 0u) {
          uint32_t slot = 0u;
          if (lane == 0) slot = atomicAdd(&s.pool, unsigned(__popc(ones)));
          slot = __shfl_sync(0xffffffffu, slot, 0) + unsigned(__popc(ones & ((1u << lane) - 1u)));
          if (single && slot < kPool)
            s.candidates[slot] = (uint64_t(s.group_keys[i]) << 32) |
                                 (0xffffffffu - (g * uint32_t(kGroup) + (word[j] >> 27)));
        }
        const unsigned rest = ~ones;
        if (rest != 0u) {
          uint32_t slot = 0u;
          if (lane == 0) slot = atomicAdd(&s.multi, unsigned(__popc(rest)));
          slot = __shfl_sync(0xffffffffu, slot, 0) + unsigned(__popc(rest & ((1u << lane) - 1u)));
          if (!single && slot < kTopK) s.multi_groups[slot] = g;
        }
      }
    }
  }
  __syncthreads();
  const int nmulti = int(s.multi);
  // Rebuild every other selected group's items from the FP32 scores.
  for (int g0 = warp; g0 < nmulti; g0 += kWarps * 16) {
    int ids[16];
    float x[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int g = g0 + j * kWarps;
      ids[j] = g < nmulti ? int(s.multi_groups[g]) * kGroup + lane : rows;
      x[j] = ids[j] < rows ? __ldcg(scores + ids[j]) : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint64_t value = composite(x[j], uint32_t(ids[j]));
      const bool keep = ids[j] < rows && uint32_t(value >> 32) >= floor_key;
      const unsigned ballot = __ballot_sync(0xffffffffu, keep);
      if (ballot != 0u) {
        uint32_t slot = 0u;
        if (lane == 0) slot = atomicAdd(&s.pool, unsigned(__popc(ballot)));
        slot = __shfl_sync(0xffffffffu, slot, 0) + unsigned(__popc(ballot & ((1u << lane) - 1u)));
        if (keep && slot < kPool) s.candidates[slot] = value;
      }
    }
  }
  __syncthreads();
  const int pool = int(s.pool);
  if (pool > kPool) {
    if (tid == 0) {
      s.prefix = 0;
      s.mask = 0;
    }
    __syncthreads();
    const uint64_t cutoff = kth(ScoreRow{scores}, rows, s);
    emit(ScoreRow{scores}, rows, cutoff, out, s);
    return;
  }
  if (pool < kTopK) __trap();
  if (tid == 0) {
    // Every pool key lies in [floor_key, key_max]: leading bytes common to
    // both bounds are common to every pool composite.
    const uint32_t diff = floor_key ^ key_max;
    const int common = diff == 0u ? 4 : __clz(int(diff)) / 8;
    const uint64_t lead = common == 0 ? uint64_t(0) : ~uint64_t(0) << (64 - 8 * common);
    s.mask = lead;
    s.prefix = (uint64_t(key_max) << 32) & lead;
  }
  __syncthreads();
  const uint64_t cutoff = pool == kTopK ? uint64_t(0) : kth(CandidateRow{s.candidates}, pool, s);
  emit(CandidateRow{s.candidates}, pool, cutoff, out, s);
}
}  // namespace flashinfer::glm5_tp8::indexer_q4::long_topk
