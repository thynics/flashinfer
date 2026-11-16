/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define MOE_HD __host__ __device__
#else
#define MOE_HD
#endif

namespace flashinfer::glm5_tp8::moe_q1 {

constexpr int kGrid = 148;
constexpr int kConsumerThreads = 256;
constexpr int kConsumerWarps = 8;
constexpr int kRoutedCtas = 128;
constexpr int kSharedBase = 128;
constexpr int kSharedCtas = 16;
constexpr int kWorkCtas = kRoutedCtas + kSharedCtas;
constexpr int kSentinelHelperCtas = kGrid - kWorkCtas;
constexpr int kWorld = 8;

constexpr int kSeqLen = 4;
static_assert(kSentinelHelperCtas == kSeqLen, "one sentinel helper CTA per token");
constexpr int kDim = 6144;
constexpr int kDimWords = kDim / 2;
constexpr int kWordsPerThread = kDimWords / kConsumerThreads;
constexpr int kTopk = 8;
constexpr int kExpertSlots = 1 + kTopk;
constexpr int kInterPerRank = 256;
constexpr int kRowsPerGroup = 16;
constexpr int kRowGroupsPerCta = 4;
constexpr int kUpGatePages = 6;
constexpr int kDownRowsPerCta = 48;
constexpr int kDownSlices = 128;
constexpr int kDownBlocksPerCta = 4;
constexpr int kRouterPages = 4;
constexpr float kRouteScale = 2.5f;
constexpr float kRmsEps = 1e-5f;
constexpr float kRmsDenominator = 6144.0f;

constexpr uint32_t kGammaPageBytes = 0x6000;
constexpr uint32_t kRouterPageBytes = 0x6000;
constexpr uint32_t kUpGatePageStride = 0x8080;
constexpr uint32_t kUpGatePageCopyBytes = 0x8040;
constexpr uint32_t kUpGateRowGroupStride = 0x30300;
constexpr uint32_t kUpGateExpertStride = 0x303000;
constexpr uint32_t kUpGateUpHalf = 0x4000;
constexpr uint32_t kUpGateScaleGate = 0x8000;
constexpr uint32_t kUpGateScaleUp = 0x8020;
constexpr uint32_t kUpGateWarpKBytes = 0x800;
constexpr uint32_t kUpGateKStepBytes = 0x100;
constexpr uint32_t kDownExpertStride = 0x180000;
constexpr uint32_t kDownSliceBytes = 0x3000;
constexpr uint32_t kDownScaleExpertStride = 0x2000;
constexpr uint32_t kDownScaleSliceBytes = 0x40;
constexpr uint32_t kDownTileBytes = 0x1000;
constexpr uint32_t kDownWarpKBytes = 0x200;
constexpr int kDownHalfBlocks = 2;
constexpr int kDownHalfGroups = kDownBlocksPerCta / kDownHalfBlocks;
constexpr uint32_t kDownHalfMatBytes = kDownHalfBlocks * kDownSliceBytes;
constexpr uint32_t kDownHalfScaleBytes = kDownHalfBlocks * kDownScaleSliceBytes;
constexpr uint32_t kDownHalfStageBytes = kDownHalfMatBytes + kDownHalfScaleBytes;
constexpr int kDownHalfStages = kDownHalfGroups * kExpertSlots;
static_assert(kDownHalfGroups == 2 && kDownHalfMatBytes == 0x6000 && kDownHalfScaleBytes == 0x80 &&
                  kDownHalfStageBytes == 0x6080,
              "two-slice expert-major down stage geometry");
constexpr int kRoutedItems = 1 + kRouterPages + kRowGroupsPerCta * kUpGatePages + kDownHalfStages;
constexpr int kSharedItems = 1 + kUpGatePages;
constexpr uint32_t kRoutedStreamBytes = kGammaPageBytes + kRouterPages * kRouterPageBytes +
                                        kRowGroupsPerCta * kUpGatePages * kUpGatePageCopyBytes +
                                        kDownHalfStages * kDownHalfStageBytes;
constexpr uint32_t kSharedStreamBytes = kGammaPageBytes + kUpGatePages * kUpGatePageCopyBytes;
static_assert(kRoutedItems == 47 && kSharedItems == 7, "item sequences");
static_assert(kGammaPageBytes % 64 == 0 && kRouterPageBytes % 64 == 0 &&
                  kUpGatePageCopyBytes % 64 == 0,
              "single-segment pages split into 4 x 16 B-multiples (0)");
static_assert(kRoutedStreamBytes == 1355520u && kSharedStreamBytes == 221568u, "bytes per CTA");

constexpr uint32_t kScoresSlotBytes = 128;
constexpr uint32_t kXRowBytes = kDim * 2;

constexpr uint32_t kInboxTokenStride = 0x38000;
constexpr uint32_t kInboxSlotStride = 0x8000;
constexpr uint32_t kInboxParityStride = 0x4000;
constexpr uint32_t kInboxChunkStride = 0x80;
constexpr uint32_t kInboxLineStride = 0x10;

static_assert(kInboxTokenStride == 7 * kInboxSlotStride &&
                  kInboxSlotStride == 2 * kInboxParityStride &&
                  kInboxParityStride == kDownSlices * kInboxChunkStride,
              "inbox geometry");

constexpr uint32_t kW1Arrivers = kSharedCtas;
MOE_HD constexpr uint32_t w1_target(uint32_t epoch) { return kW1Arrivers * epoch; }
MOE_HD constexpr uint32_t init_target(uint32_t epoch) { return kSentinelHelperCtas * epoch; }

struct CtxRow0 {
  const uint16_t* x;
  const float* gamma;
  uint64_t pad[14];
};
struct CtxRow1 {
  const uint16_t* router_w;
  float* scores_out;
  uint64_t pad[14];
};
struct CtxRow2 {
  const uint8_t* upgate_w;
  const float* bias;
  const float* scores_in;
  float* sel_probs;
  int32_t* sel_indices;
  uint16_t* hidden_mid_out;
  uint64_t pad[10];
};
struct CtxRow3 {
  const uint8_t* upgate_w_shared;
  uint16_t* hidden_mid_shared_out;
  uint16_t* x_norm_buf;
  const uint16_t* x_shared;
  uint64_t pad[12];
};
struct CtxRow4 {
  const uint8_t* down_w;
  const uint16_t* hidden_mid_in;
  uint16_t* hidden_out;
  const uint16_t* x_residual;
  uint64_t epoch;
  const float* down_scale;
  uint64_t pad[10];
};
struct CtxTable {
  CtxRow0 r0;
  CtxRow1 r1;
  CtxRow2 r2;
  CtxRow3 r3;
  CtxRow4 r4;
};
constexpr int kCtxRows = 5;
constexpr int kCtxRowBytes = 128;
static_assert(sizeof(CtxRow0) == kCtxRowBytes && sizeof(CtxRow1) == kCtxRowBytes &&
                  sizeof(CtxRow2) == kCtxRowBytes && sizeof(CtxRow3) == kCtxRowBytes &&
                  sizeof(CtxRow4) == kCtxRowBytes && sizeof(CtxTable) == kCtxRows * kCtxRowBytes,
              "ctx rows are 128 B");
static_assert(offsetof(CtxRow0, gamma) == 0x08, "row 0");
static_assert(offsetof(CtxRow1, scores_out) == 0x08, "row 1");
static_assert(offsetof(CtxRow2, bias) == 0x08 && offsetof(CtxRow2, scores_in) == 0x10 &&
                  offsetof(CtxRow2, sel_probs) == 0x18 && offsetof(CtxRow2, sel_indices) == 0x20 &&
                  offsetof(CtxRow2, hidden_mid_out) == 0x28,
              "row 2");
static_assert(offsetof(CtxRow3, hidden_mid_shared_out) == 0x08 &&
                  offsetof(CtxRow3, x_norm_buf) == 0x10 && offsetof(CtxRow3, x_shared) == 0x18,
              "row 3");
static_assert(offsetof(CtxRow4, hidden_mid_in) == 0x08 && offsetof(CtxRow4, hidden_out) == 0x10 &&
                  offsetof(CtxRow4, x_residual) == 0x18 && offsetof(CtxRow4, epoch) == 0x20 &&
                  offsetof(CtxRow4, down_scale) == 0x28,
              "row 4");
static_assert(offsetof(CtxTable, r1) == 0x80 && offsetof(CtxTable, r2) == 0x100 &&
                  offsetof(CtxTable, r3) == 0x180 && offsetof(CtxTable, r4) == 0x200,
              "ctx table rows");

constexpr int kQ1Block = 384;
constexpr uint32_t kQ1DynamicSmem = 214016;
struct LaunchArgs {
  uint32_t n_rows;
  int32_t rank;
  const uint8_t* context;
  uint32_t* sync_words;
  uint8_t* peers[8];
  uint32_t* launch_ctr;
  uint32_t* reset_token;
};
static_assert(sizeof(LaunchArgs) == 104, "q1 launch layout");
}  // namespace flashinfer::glm5_tp8::moe_q1
