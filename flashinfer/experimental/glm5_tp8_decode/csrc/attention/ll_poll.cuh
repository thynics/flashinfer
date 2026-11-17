/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define LL_POLL_HD __host__ __device__
#else
#define LL_POLL_HD
#endif

namespace ll_wire {

// Check the latest loaded tags, including the final allowed reload. This pure
// predicate is shared with the CPU boundary regression; production calls it
// only inside the existing cold deadline branch.
LL_POLL_HD constexpr bool deadline_missing_tags(uint32_t polls, uint32_t limit, uint32_t tag0,
                                                uint32_t tag1, uint32_t expected) {
  return polls >= limit && (tag0 != expected || tag1 != expected);
}

#if defined(__CUDACC__)
// A missing publication is an execution error. Called only at the existing
// production timeout branches; keep the hot poll/consume code at its old sites.
__device__ __forceinline__ void timeout_trap() { __trap(); }
#endif

}  // namespace ll_wire

#undef LL_POLL_HD
