/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// A long row never enters the legacy 18-bit, epoch-tagged selector. P4
// publishes complete scores/summaries through C4 before this scratch is read.
namespace flashinfer::glm5_tp8::indexer_q4::long_topk {
constexpr int kTopK = 2048;
constexpr int kGroup = 8;
constexpr int kPool = kTopK * kGroup;
constexpr int kThreads = 384;
constexpr int kWarps = kThreads / 32;
constexpr int kLogicalMax = 1048576 + 4096;

__host__ __device__ constexpr int group_count(int length) { return (length + kGroup - 1) / kGroup; }
__host__ __device__ constexpr int summary_stride(int length) {
  return (group_count(length) + 1) & ~1;
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
// has exactly one writer; release/acquire at C4 makes atomic publication tags
// unnecessary. Never preload these summaries before C4.
__device__ __forceinline__ void publish_group(float score, int position, int valid_rows,
                                              int physical_rows, uint64_t* summary) {
  uint64_t key = position < valid_rows ? composite(score, uint32_t(position)) : 0;
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    const uint64_t other = warp_down(key, offset, kGroup);
    key = key > other ? key : other;
  }
  if ((threadIdx.x & (kGroup - 1)) == 0 && position < physical_rows)
    store_cg(summary + position / kGroup, key);
}

struct alignas(16) Scratch {
  uint64_t candidates[kPool];
  uint32_t groups[kTopK];
  uint32_t warp_hist[kWarps][256];
  uint32_t hist[256];
  uint32_t count;
  uint32_t remaining;
  uint32_t bucket_count;
  uint32_t failed;
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
template <class Row>
__device__ __forceinline__ uint64_t kth(Row row, int count, Scratch& s) {
  const int tid = int(threadIdx.x), lane = tid & 31, warp = tid >> 5;
  if (tid == 0) {
    s.prefix = 0;
    s.mask = 0;
    s.remaining = kTopK;
    s.failed = 0;
  }
  __syncthreads();
  for (int shift = 56; shift >= 0; shift -= 8) {
    for (int i = tid; i < kWarps * 256; i += kThreads)
      reinterpret_cast<uint32_t*>(s.warp_hist)[i] = 0;
    __syncthreads();
    const uint64_t prefix = s.prefix, mask = s.mask;
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
    if (tid == 0) {
      uint32_t remaining = s.remaining;
      bool found = false;
      for (int digit = 255; digit >= 0; --digit) {
        const uint32_t n = s.hist[digit];
        if (remaining > n) {
          remaining -= n;
        } else {
          s.prefix |= uint64_t(digit) << shift;
          s.mask |= uint64_t(255) << shift;
          s.remaining = remaining;
          s.bucket_count = n;
          found = true;
          break;
        }
      }
      if (!found) s.failed = 1;
    }
    __syncthreads();
    if (s.failed) __trap();
    // All keys with a valid ID are unique. Find the sole remaining key rather
    // than rereading the entire row for its otherwise irrelevant suffix bytes.
    if (s.bucket_count == 1) {
      for (int i = tid; i < count; i += kThreads) {
        const uint64_t value = row(i);
        if ((value & s.mask) == s.prefix) s.cutoff = value;
      }
      __syncthreads();
      return s.cutoff;
    }
  }
  return s.prefix;
}

template <class Row>
__device__ __forceinline__ void emit(Row row, int count, uint64_t cutoff, int32_t* out,
                                     Scratch& s) {
  const int tid = int(threadIdx.x);
  if (tid == 0) s.count = 0;
  __syncthreads();
  for (int i = tid; i < count; i += kThreads) {
    const uint64_t value = row(i);
    if (value >= cutoff) {
      const uint32_t slot = atomicAdd(&s.count, 1u);
      if (slot < kTopK) out[slot] = int32_t(0xffffffffu - uint32_t(value));
    }
  }
  __syncthreads();
  if (s.count != kTopK) __trap();
}

// Keep the large radix body outside the persistent kernel's short-row path.
static __device__ __noinline__ void select(const float* scores, int rows, const uint64_t* maxima,
                                           int32_t* out, void* scratch, bool grouped) {
  Scratch& s = *reinterpret_cast<Scratch*>(scratch);
  const int tid = int(threadIdx.x);
  if (rows <= kTopK) {
    for (int i = tid; i < kTopK; i += kThreads) out[i] = i < rows ? i : -1;
    __syncthreads();
    return;
  }
  if (!grouped || group_count(rows) < kTopK) {
    const uint64_t cutoff = kth(ScoreRow{scores}, rows, s);
    emit(ScoreRow{scores}, rows, cutoff, out, s);
    return;
  }
  const int ngroups = group_count(rows);
  const uint64_t threshold = kth(SummaryRow{maxima}, ngroups, s);
  if (tid == 0) s.count = 0;
  __syncthreads();
  for (int i = tid; i < ngroups; i += kThreads) {
    if (load_cg(maxima + i) >= threshold) {
      const uint32_t slot = atomicAdd(&s.count, 1u);
      if (slot < kTopK) s.groups[slot] = uint32_t(i);
    }
  }
  __syncthreads();
  if (s.count != kTopK) __trap();
  // Unique composite maxima select exactly K disjoint groups. Their union
  // contains every global top-K item and has at most K*8 entries, even ties.
  for (int i = tid; i < kPool; i += kThreads) {
    const int id = int(s.groups[i / kGroup]) * kGroup + i % kGroup;
    s.candidates[i] = id < rows ? composite(load_cg(scores + id), uint32_t(id)) : 0;
  }
  __syncthreads();
  const uint64_t cutoff = kth(CandidateRow{s.candidates}, kPool, s);
  emit(CandidateRow{s.candidates}, kPool, cutoff, out, s);
}
}  // namespace flashinfer::glm5_tp8::indexer_q4::long_topk
