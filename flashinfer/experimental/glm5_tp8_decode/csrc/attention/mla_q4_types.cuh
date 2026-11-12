/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define MEGA_ATTN_HD __host__ __device__
#else
#define MEGA_ATTN_HD
#endif

#ifndef MEGA_ATTN_LEGACY_GEMV
#define MEGA_ATTN_LEGACY_GEMV 0
#endif

#ifndef MEGA_ATTN_WAIT_SLEEP_NS
#define MEGA_ATTN_WAIT_SLEEP_NS 0
#endif

namespace flashinfer::glm5_tp8::mla_q4 {

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
constexpr int kWorld = 8;
constexpr int kGrid = 148;                 // persistent grid = CTAs = SMs on B200
constexpr int kBlock = 384;                // 8 consumer warps + 4 producer warps
constexpr int kOpWarps = 8;                // consumer warps that execute op bodies
constexpr int kOpThreads = kOpWarps * 32;  // 256
constexpr int kProducerWarps = 4;
constexpr int kProducerThreads = kProducerWarps * 32;  // 128
constexpr int kSourceBlock = 384;
constexpr int kQlen = 4;
constexpr int kHidden = 6144;
constexpr int kQlora = 2048;
constexpr int kKvlora = 512;
constexpr int kNope = 192;
constexpr int kRope = 64;
constexpr int kHeads = 10;
constexpr int kVdim = 256;
constexpr int kSelected = 2048;

constexpr int kQkvaRows = kQlora + kKvlora + kRope;  // 2624
constexpr int kQkvaTiles = kQkvaRows / 16;           // 164 (16-row tiles)
constexpr int kQkvaQTiles = kQlora / 16;             // 128
constexpr int kQkvaKvTiles = kKvlora / 16;           // 32
constexpr int kQkvaPeTiles = kRope / 16;             // 4
constexpr int kQkvaCtaBlocks = 82;                   // 32-row blocks (K1 CTA c < 82 owns block c)
constexpr int kQkvaQBlocks = kQlora / 32;            // 64  (CTAs 0..63  -> post_q)
constexpr int kQkvaKvBlocks = kKvlora / 32;          // 16  (CTAs 64..79 -> post_kv)
constexpr int kQkvaPeBlocks = kRope / 32;            // 2   (CTAs 80..81 -> pe_raw)
constexpr int kQkvaPagesPerBlock = 6;
static_assert(kQkvaQBlocks + kQkvaKvBlocks + kQkvaPeBlocks == kQkvaCtaBlocks, "QKVA block split");
constexpr int kPageBytes8080 = 0x8080;  // QKVA / WqB page
constexpr int kProjqPageBytes = 0x3080;
constexpr int kProjoPageBytes = 0x4080;
constexpr int kUnprojPageBytes = 0x6080;
constexpr int kUnprojPageBulkBytes =
    0x6030;  // bytes the ring copies per UnProj page (matrix + 12 scales)
constexpr int kWqbRows = kHeads * (kNope + kRope);  // 2560
constexpr int kWqbNopeRows = kHeads * kNope;        // 1920
constexpr int kWqbPeRows = kHeads * kRope;          // 640
constexpr int kWqbTiles = kWqbRows / 16;            // 160
constexpr int kWqbCtas = 80;                        // 32 rows each, 2 pages
constexpr int kProjqRows = kHeads * kKvlora;        // 5120
constexpr int kProjqCtas = 80;                      // 64 rows each, 1 page
constexpr int kProjqTileRows = 18;
constexpr int kProjoRows = kHeads * kVdim;  // 2560
constexpr int kProjoCtas = 80;              // 32 rows each, 1 page
constexpr int kUnprojCtas = 128;            // 48 rows each, 5 pages
constexpr int kUnprojRowsPerCta = 48;
constexpr int kUnprojTilesPerCta = 3;
constexpr size_t kQkvaBytes = 16184832;
constexpr size_t kWqbBytes = 5263360;
constexpr size_t kProjqBytes = 993280;
constexpr size_t kProjoBytes = 1320960;
constexpr size_t kUnprojBytes = 15810560;
static_assert(kQkvaBytes == size_t(kQkvaCtaBlocks) * kQkvaPagesPerBlock * kPageBytes8080,
              "qkva blob");
static_assert(kWqbBytes == size_t(kWqbCtas) * 2 * kPageBytes8080, "wqb blob");
static_assert(kProjqBytes == size_t(kProjqCtas) * kProjqPageBytes, "projq blob");
static_assert(kProjoBytes == size_t(kProjoCtas) * kProjoPageBytes, "projo blob");
static_assert(kUnprojBytes == size_t(kUnprojCtas) * 5 * kUnprojPageBytes, "unproj blob");
static_assert(kUnprojPageBulkBytes % 16 == 0 && kUnprojPageBulkBytes >= 0x6000 + 12 * 4 &&
                  kUnprojPageBulkBytes <= kUnprojPageBytes,
              "UnProj bulk bytes cover the matrix + scales, 16 B multiple");

constexpr int kScoreSlots = 36;
constexpr int kScoreSplits = (kSelected + kScoreSlots - 1) / kScoreSlots;  // 57
constexpr int kAttnStride = 584;                                           // 512 + 64 + 8 pad
constexpr int kPvDimTiles = kKvlora / 16;                                  // 32
constexpr int kPvHeadGroups = (kHeads + 7) / 8;                            // 2
constexpr int kPvWarps = kQlen * kPvHeadGroups * kPvDimTiles;              // 256

// ---------------------------------------------------------------------------
// Op ids and CTA sets.  Group1 = CTAs 0..127 runs the phases QKVA -> WqB -> ProjQ -> attention ->
// ProjO -> UnProj + collective; the 20 tail CTAs 128..147 run kv-norm (128..131), pe RoPE
// (132..135), the selected-id copy slices (136..143) and otherwise idle, arrive C3 / C2 and exit
// (the attention phase is Group1's alone).
// ---------------------------------------------------------------------------
enum OpId : int {
  kOpK0 = 0,        // rank0 publish + wait (rank0 only)                     CTA set: 148
  kOpK1 = 1,        // x-norm + QKVA GEMV (block c for c < 82)               CTA set: 128 (Group1)
  kOpK1Tails = 2,   // kv-norm rows (tails 0..3), pe RoPE rows (4..7), ids   CTA set: 8 (wrapper) /
                    // 20 (persistent)
  kOpK2Wqb = 3,     // q-norm + WqB (+RoPE)                                  CTA set: 80
  kOpK2ProjQ = 4,   // ProjQ                                                 CTA set: 80
  kOpK3 = 5,        // scores + softmax + PV                                 CTA set: 148
  kOpK4 = 6,        // (combine) + ProjO                                     CTA set: 80
  kOpK5 = 7,        // Unproj + TP7 collective + residual + done             CTA set: 128
  kOpK5Unproj = 8,  // K5 phase A only (loopback / per-op tuning)
  kOpK5Collective = 9,  // K5 phase B only (loopback / per-op tuning)
  kOpK5UnprojOnly = 10,
  // per-op tuning): no arena mirror, no peers, no collective, no flags, no done
  kOpK2Fused = 11,  // fused q-norm + WqB half tiles + page-local ProjQ (WqB + ProjQ)   CTA set: 120
  kOpCount = 12,
};
constexpr int kGroup1Ctas = 128;
constexpr int kTailCtas = kGrid - kGroup1Ctas;  // 20 tail CTAs 128..147
constexpr int kKvNormCtas = kQlen;              // tails 0..3  = CTAs 128..131: kv-norm of row t
constexpr int kRopeCtas =
    kTailCtas - kKvNormCtas;      // tails 4..19 = CTAs 132..147 (C2 group; 4..7 work)
constexpr int kTailRope = kQlen;  // first RoPE tail index
constexpr int kTailIds =
    2 * kQlen;  // tail 8 = CTA 136: first selected-id copy CTA (persistent kernel only)
constexpr int kIdsCopyCtas =
    8;  // tails 8..15 = CTAs 136..143 each pull one coalesced 4 KiB ids slice
constexpr int kK0Ctas = kGrid;
constexpr int kK1Ctas = kGroup1Ctas;     // 82 own QKVA pages, the rest idle in K1
constexpr int kK1TailsCtas = 2 * kQlen;  // standalone wrapper: kv-norm + RoPE rows (chain mode)
constexpr int kK2Ctas = 80;
constexpr int kK3Ctas = kGrid;
constexpr int kK4Ctas = 80;
constexpr int kK5Ctas = 128;
//  fused K2 geometry: 10 heads x 12 cohort CTAs.  Cohort k of
// head h computes the 16-row WqB NOPE half tile rows [192h + 16k, +16) (row half k & 1 of 32-row
// block 6h + k/2; both 0x8080 pages of the block are streamed, one half is used) and arrives on
// kCntK2Head + h; cohorts 0..7 then run ProjQ rows [64(8h + k), +64) after the head's 12 arrivals
// (page-local hand-off, no grid-wide C1 boundary); cohorts 8..11 compute the PE half tiles rows
// [1920 + 64h + 16(k-8), +16) (block 60 + 2h + (k-8)/2).  CTAs 120..127 of Group1 idle in this
// phase.
constexpr int kK2FusedCohort = 12;
constexpr int kK2FusedProjqPerHead = 8;
constexpr int kK2FusedCtas = kHeads * kK2FusedCohort;  // 120
constexpr int kK4FusedPages = 4;
static_assert(
    kK2FusedCohort * 16 == kNope && (kK2FusedCohort - kK2FusedProjqPerHead) * 16 == kRope &&
        kK2FusedProjqPerHead * 64 == kKvlora,
    "fused K2 cohort geometry (12 NOPE half tiles, 4 PE half tiles, 8 ProjQ tiles per head)");
static_assert(kK2FusedCtas <= kGroup1Ctas, "fused K2 CTAs must lie inside Group1");

constexpr int kK3OptSplits = 32;
constexpr int kK3OptRows = 64;
constexpr int kK3OptHalfRows = 32;
constexpr int kK3OptRingRowStride = kKvlora + 8;  // 520 bf16 = 1040 B per gathered KV row
constexpr int kK3SplitCtas =
    kQlen * kK3OptSplits;  // 128 (K3 CTAs 128..147 idle: the persistent tails never enter K3)
constexpr int kK3HeadsPadded = 16;
constexpr int kK3CombineCtas = 2 * kHeads;  // splits 0..19 of each query combine
static_assert(kK3SplitCtas <= kK3Ctas && kK3CombineCtas <= kK3OptSplits, "split-KV CTA geometry");
static_assert(kK2Ctas == kWqbCtas && kK2Ctas == kProjqCtas, "K2 geometry");
static_assert(kK4Ctas == kProjoCtas, "K4 geometry");
static_assert(2 * kK4FusedPages * 32 == kVdim && kK4Ctas == kHeads * kQlen * 2,
              "fused K4: (head, query, row half) CTAs, four 32-row ProjO pages each");
static_assert(kK5Ctas == kUnprojCtas, "K5 geometry");
static_assert(kK3Ctas >= kQlen * kHeads, "softmax roles need >= 40 CTAs");
static_assert(kQkvaCtaBlocks <= kK1Ctas && kK2Ctas <= kGroup1Ctas && kK4Ctas <= kGroup1Ctas &&
                  kK5Ctas == kGroup1Ctas,
              "every GEMV CTA set lies inside Group1");
static_assert(
    kKvNormCtas + kRopeCtas == kTailCtas && kTailIds + kIdsCopyCtas <= kTailCtas &&
        kK1TailsCtas <= kTailIds,
    "tail group split (4 kv-norm + 16 C2-group CTAs, the eight ids-copy CTAs inside the C2 group)");

MEGA_ATTN_HD constexpr int op_cta_count(int op) {
  return op == kOpK0                            ? kK0Ctas
         : op == kOpK1                          ? kK1Ctas
         : op == kOpK1Tails                     ? kK1TailsCtas
         : (op == kOpK2Wqb || op == kOpK2ProjQ) ? kK2Ctas
         : op == kOpK2Fused                     ? kK2FusedCtas
         : op == kOpK3                          ? kK3Ctas
         : op == kOpK4                          ? kK4Ctas
                       : kK5Ctas;  // kOpK5, kOpK5Unproj, kOpK5Collective, kOpK5UnprojOnly
}

// Execution modes (MegaArgs::mode; env MEGA_ATTN_MODE)
enum Mode : int {
  kModePersistent = 0,         // one persistent kernel per pure rank
  kModeChain = 1,              // standalone wrappers launched in stream order (same math)
  kModeNullOrchestration = 2,  // counters/flags/ids/handshake only; no math, outputs zero
  kModeNullStream = 3,         // producer warps stream every weight page; consumers idle
};

// K5 collective variants (MegaArgs::no_collective; env MEGA_ATTN_NO_COLLECTIVE, single-rank only).
// Orthogonal to the mode: persistent / chain / null_orchestration all honour it.
enum NoCollective : int {
  kCollectiveFull = 0,   //  push collective: inbox push -> own-inbox poll -> rank-ordered sum ->
                         //  K5Done (last arriver -> kFlagDone)
  kCollectiveLocal = 1,  // single GPU, no TP all-reduce: endpoint = BF16(x + BF16(local_hidden of
                         // this rank)); no K5 words / flags
  kCollectiveUnprojOnly =
      2,  // UnProj into arena::kLocalHidden only; the endpoint is not written (pure compute)
};

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
struct MegaArgs {
  int rank;
  uint32_t epoch;          // host-incremented, starts at 1, never 0
  uint8_t* peers[kWorld];  // IPC-mapped per-rank arenas (peers[rank] == local)
  const __nv_bfloat16* x;
  const float* x_gamma;
  const float* q_gamma;
  const float* kv_gamma;
  const float* rope;
  const uint8_t* qkva;
  const uint8_t* wqb;
  const uint8_t* projq;
  const uint8_t* projo;
  const uint8_t* unproj;
  __nv_bfloat16* kv_cache;
  __nv_bfloat16* pe_cache;
  const int32_t* selected_ids_src;
  const int32_t* selected_lengths_src;
  int cur_pos;
  int cache_rows;
  float x_eps;
  float q_eps;
  float kv_eps;
  float softmax_scale;
  __nv_bfloat16* post_q;
  __nv_bfloat16* post_kv;
  __nv_bfloat16* post_qnd;
  __nv_bfloat16* post_qpe;
  __nv_bfloat16* post_qnope;
  __nv_bfloat16* post_o;
  __nv_bfloat16* post_projo;
  __nv_bfloat16* endpoint;
  // implementation additions
  int mode;        // Mode
  int ids_direct;  // 1: K3 copies ids/lengths from selected_ids_src/lengths_src

  int skip_done_wait;  // 1: K0 (rank0) skips its ack/done waits.  Since the  push collective K5
                       //    never waits for peers' done (kFlagDone is a pacing token for rank0
                       //    only), so the flag is irrelevant to K5; kept for the harness bindings
  int single_rank;     // 1: single-rank loopback (env MEGA_ATTN_SINGLE_RANK=r):
                       //    peers[0..7] all alias the local arena; K0 is a
                       //    publish-only kernel launched before the op chain
                       //    (no output zeroing, no ack/done waits); K5 handshakes
                       //    with itself, so the endpoint is
                       //    BF16(x + BF16(sum of 7 copies of local_hidden)),
                       //    NOT the contract value
  int ids_ready;       // 1: the selected ids/lengths are consumed into the local
                       //    arena by the tail CTAs 136..143 before the C2 boundary
                       //    (persistent kernel), so K3 skips its own S0 consume;
                       //    0: K3 CTA 0 consumes them (chain / standalone K3)
  __nv_bfloat16* local_hidden_out;  // kOpK5UnprojOnly only: destination bf16[4][6144] of the

  //    persistent kernel / K5 phases write arena::kLocalHidden
  //    instead and ignore this pointer (may be nullptr there).
  int no_collective;  // NoCollective: 0 = the TP collective (push); 1 = local endpoint
                      //    BF16(x + BF16(local_hidden)), no K5 words / sys-scope flags /
                      //    peer waits; 2 = UnProj only, endpoint untouched.  Single-rank
                      //    loopback only (host rejects it otherwise: rank0's K0 would wait
                      //    for kFlagDone words nobody publishes).
  // --- integration additions: device-resident overrides, appended at the end so every
  // existing field offset is unchanged.  nullptr = ignored (host values above are used, exactly as
  // before).
  const int32_t* cur_pos_dev;
  const uint32_t* epoch_dev;  // if set: epoch = atomicAdd(*epoch_dev, 1) / kGrid + 1 at kernel
                              // entry (launch ordinal on this device)
  uint64_t reserved_trace;
  int rank0_reduce;  // replace mode: pure ranks also push their K5 slice into rank 0's inbox (slot
                     // rank-1); rank 0's CTAs 0..127 receive the seven slices after the publish and
                     // write endpoint = BF16(x + BF16(rank-ordered sum))
  uint64_t reserved_fill_f32;
  uint64_t reserved_fill_bf16;
  int dense_ids;  // replace mode (ctx <= kSelected): the ids tail CTAs write ids[q][i] = i and
                  // lengths[q] = min(kSelected, cur_pos+q+1) into selected_ids_src /
                  // selected_lengths_src before arriving on C2 (no host prep kernel); rank 0 skips
                  // the publish
  uint32_t* epoch_ctr;  // if set: per-CTA launch counters [kGrid]; epoch = ++epoch_ctr[cta] (plain
                        // load/store, no contended atomic);
  const uint8_t* ll_recv;  // explicit selected-ID receive buffer
  uint64_t reserved_inbox0;
  int reserved_rank0;
  int ll_ids;  // 1: for cur_pos+q+1 > kSelected consume the NCCL-LL ids rows that rank 0 pushes
               // into recv_buf
  uint64_t reserved_own_inbox;
  // slot-0 block of this layer (line 7 tag == epoch) before finishing (per-layer pacing on rank 0)
  uint32_t ll_tag;  // selected-ID protocol tag
  uint32_t reserved_wire_epoch;
};

// ---------------------------------------------------------------------------
// Global arena (per rank, one cudaMalloc, IPC-exported).  Offsets in bytes.

// ---------------------------------------------------------------------------
namespace arena {
constexpr size_t kFlagPub = 0;
constexpr size_t kFlagAck = 4;
constexpr size_t kFlagPartial = 8;
constexpr size_t kFlagDone = 12;
constexpr size_t kFlagCollective = 16;
constexpr size_t kFlagQnorm = 32;
constexpr size_t kFlagKvnorm = 48;
constexpr size_t kFlagPeCache = 64;
constexpr size_t kFlagsBytes = 128;
constexpr size_t kCounters = 128;  // 32 x u32 monotonic ArrivalCounters (implementation)
constexpr size_t kCountersBytes = 128;
constexpr size_t kLengths = 256;
constexpr size_t kLengthsBytes = 256;
constexpr size_t kIds = 512;
constexpr size_t kIdsBytes = size_t(kQlen) * kSelected * 4;  // 32768
constexpr size_t kXnorm = 33280;
constexpr size_t kXnormBytes = size_t(kQlen) * kHidden * 2;  // 49152
constexpr size_t kPeRaw = 82432;
constexpr size_t kPeRawBytes = size_t(kQlen) * kRope * 2;  // 512
constexpr size_t kQnorm = 82944;
constexpr size_t kQnormBytes = size_t(kQlen) * kQlora * 2;  // 16384
constexpr size_t kScores = 99328;
constexpr size_t kScoresBytes = size_t(kQlen) * kHeads * kSelected * 4;  // 327680
constexpr size_t kProbs = 427008;
constexpr size_t kProbsBytes = size_t(kQlen) * kHeads * kSelected * 2;  // 163840

constexpr size_t kCountersPadded = kProbs;  // 32 x 128 B lines in the unused probs region
constexpr size_t kCountersPaddedBytes = 4096;
static_assert(kCountersPaddedBytes <= kProbsBytes, "padded counters fit the dead probs region");
// K1-published q-norm sum-of-square partials, f32 [q block][token].
// Every q-block CTA writes one aligned float4 token record before releasing C0, so
// fused K2 readers inherit visibility from their existing C0 acquire.
constexpr size_t kQnormPartial = kProbs + kCountersPaddedBytes;
constexpr size_t kQnormPartialBytes = size_t(kQlen) * kQkvaQBlocks * 4;
static_assert(kCountersPaddedBytes + kQnormPartialBytes <= kProbsBytes,
              "qnorm partials fit the dead probs region");
constexpr size_t kStats = 590848;
constexpr size_t kStatsBytes = 512;
constexpr size_t kLocalHidden = 591360;
constexpr size_t kLocalHiddenBytes = size_t(kQlen) * kHidden * 2;  // 49152

constexpr size_t kOAcc = 640512;  // f32[4][57 splits][10][512] partial PV accumulators
constexpr size_t kOAccBytes = size_t(kQlen) * kScoreSplits * kHeads * kKvlora * 4;  // 4669440

constexpr int kRxSlots = kWorld - 1;  // 7 source slots (slot r-1 = pure rank r)
constexpr int kRxRecordElems = 6;     // bf16 per record (12 B) + u32 epoch
constexpr int kRxRecordsPerToken = kUnprojRowsPerCta / kRxRecordElems;  // 8
constexpr size_t kRxRecordBytes = 16;
constexpr size_t kRxCtaBytes = size_t(kQlen) * kRxRecordsPerToken * kRxRecordBytes;  // 512
constexpr size_t kRxSlotBytes = size_t(kK5Ctas) * kRxCtaBytes;                       // 65536
constexpr size_t kRxParityBytes = size_t(kRxSlots) * kRxSlotBytes;                   // 458752
constexpr size_t kHiddenRx = 5309952;
constexpr size_t kHiddenRxBytes = size_t(2) * kRxParityBytes;  // 917504
static_assert(kUnprojRowsPerCta % kRxRecordElems == 0 && kRxRecordsPerToken * kRxRecordBytes == 128,
              "one token's 48-element slice = 8 records = one 128 B line");
// Exit credit, rank 0 -> pure rank: one 16 B line {E, E, E, E} per K5 CTA, E = the last epoch whose
// kHiddenRx records rank 0 has consumed.  Rank 0 stores it when its next q4 launch starts (every
// earlier launch on its stream has completed by then).  A CTA pushes epoch E into rank 0's slot of
// parity E & 1 only once its credit reaches E - 2, the previous use of that slot: in dense
// positions rank 0 publishes no ids, and nothing else stops back-to-back attention launches from
// overwriting unread records. Zeroed with the arena, compared modulo 2^32; placed in the ids
// region, which no path of this kernel uses.
constexpr size_t kExitCredit = kIds;
constexpr size_t kExitCreditLineBytes = 16;
constexpr size_t kExitCreditBytes = size_t(kK5Ctas) * kExitCreditLineBytes;
static_assert(kExitCredit == 512 && kExitCreditBytes <= kIdsBytes,
              "exit credits: wire offset of indexer_abi.h mega_wire::kExitCredit, inside the "
              "unused ids region");
constexpr size_t kEnd = kHiddenRx + kHiddenRxBytes;  // 6227456

constexpr size_t kTrace = kEnd;                                      // 6227456
constexpr int kTraceStamps = 64;                                     // TraceId slots per CTA
constexpr int kTraceEpochSlots = 16;                                 // ring of epochs kept
constexpr size_t kTraceCtaBytes = size_t(kTraceStamps) * 8;          // 512
constexpr size_t kTraceEpochBytes = size_t(kGrid) * kTraceCtaBytes;  // 75776
constexpr size_t kTraceBytes = size_t(kTraceEpochSlots) * kTraceEpochBytes;  // 1212416
// Rank 0 pushes the 32 KiB ids and four lengths into every pure rank's local,
// parity-buffered inbox.  The peer then waits on its local publication word;
// no pure-rank CTA reads rank 0's arena remotely.  Ack pacing plus per-rank
// stream order makes the two-epoch reuse interval safe.
constexpr size_t kIdsRx = kTrace + kTraceBytes;  // 7439872
constexpr size_t kIdsRxLengths = kIdsBytes;
constexpr size_t kIdsRxParityBytes = kIdsBytes + 128;          // 32896
constexpr size_t kIdsRxBytes = size_t(2) * kIdsRxParityBytes;  // 65792
constexpr size_t kArenaBytes = size_t(8) * 1024 * 1024;        // 8 MiB

// Non-overlap and 128 B alignment.
static_assert(kFlagsBytes == kCounters, "flags/counters adjacency");
static_assert(kCounters + kCountersBytes <= kLengths, "counters overlap lengths");
static_assert(kLengths + kLengthsBytes <= kIds, "lengths overlap ids");
static_assert(kIds + kIdsBytes <= kXnorm, "ids overlap xnorm");
static_assert(kXnorm + kXnormBytes <= kPeRaw, "xnorm overlap pe_raw");
static_assert(kPeRaw + kPeRawBytes <= kQnorm, "pe_raw overlap qnorm");
static_assert(kQnorm + kQnormBytes <= kScores, "qnorm overlap scores");
static_assert(kScores + kScoresBytes <= kProbs, "scores overlap probs");
static_assert(kProbs + kProbsBytes <= kStats, "probs overlap stats");
static_assert(kStats + kStatsBytes <= kLocalHidden, "stats overlap local_hidden");
static_assert(kLocalHidden + kLocalHiddenBytes <= kOAcc, "local_hidden overlap o_acc");
static_assert(kOAcc + kOAccBytes <= kHiddenRx, "o_acc overlap hidden_rx");
static_assert(kEnd <= kArenaBytes, "arena overflow");
static_assert(kTrace % 128 == 0 && kIdsRx % 128 == 0 && kIdsRx + kIdsRxBytes <= kArenaBytes,
              "appended regions overflow");
static_assert(kCounters % 128 == 0 && kLengths % 128 == 0 && kIds % 128 == 0 && kXnorm % 128 == 0 &&
                  kPeRaw % 128 == 0 && kQnorm % 128 == 0 && kScores % 128 == 0 &&
                  kProbs % 128 == 0 && kStats % 128 == 0 && kLocalHidden % 128 == 0 &&
                  kOAcc % 128 == 0 && kHiddenRx % 128 == 0,
              "arena regions must be 128 B aligned");

static_assert(kLengths == 256 && kIds == 512 && kXnorm == 33280 && kPeRaw == 82432 &&
                  kQnorm == 82944 && kScores == 99328 && kProbs == 427008 && kStats == 590848 &&
                  kLocalHidden == 591360,
              "scratch layout drift");
}  // namespace arena

// Counter / boundary-word slots (u32 index inside arena::kCounters).  Every word is monotonic and
// receives exactly `count` arrivals per epoch, so the wait target for epoch e is count*e.  For the
// boundary words C0..C7 EVERY CTA of the arrival group arrives once, working or idle, and there is
// no last-arriver /

enum CounterId : int {
  kCntK0Copy = 0,  // 1 per epoch: the publisher CTA (rank0 CTA 0 / single-rank publish kernel),
                   // bookkeeping only
  kCntIdsRx =
      1,       // 1 per epoch, only when K3 consumes the ids itself (ids_ready == 0)   K3 internal
  kCntC0 = 2,  // 82 QKVA producers after bulk-published outputs/Q-norm records -> WqB CTAs,
               // kv-norm/RoPE tails
  kCntC1 = 3,
  kCntC2 =
      4,  // 5 RoPE/ids producers (tails 4..8) after pe_cache rows / local ids acquire -> attention
  kCntC3 = 5,  // 4 = kv-norm group (tails 0..3) after kv_cache rows        -> attention
  kCntC4 = 6,  // 120 fused-K2 producers after ProjQ/PE                     -> attention
  kCntC6 = 7,  // retired: ProjO now waits on the producing head's kCntK2Head cycle
  kCntK3Partials =
      8,  // [4] 32 per epoch each (one per split CTA of the query)    K3 internal -> combine CTAs
  kCntK3Stats = 12,  // [4] 32 split statistics per epoch; global P normalization before BF16 PV
  kCntC7 = 16,       // 80 ProjO producers (post_proj_o)                          -> UnProj CTAs
  kCntC5 = 17,       // reserved: "partials landed" word of the split-KV attention (item 4), unused
  kCntC4Pe = 18,
  kCntK5Collective = 19,
  kCntIdsCopy = 20,
  kCntK2Head =
      21,  // [10] 20 per epoch: 12 K2 cohorts publish QND, then 8 K3 combine halves publish post_o
  kCntCount = 32,
};
// The counter WORDS keep their ids and meaning but each now occupies its own
// 128 B L2 line inside arena::kCountersPadded (kCntLine u32 words apart): a
// release reduction on one boundary no longer invalidates the polled line of every
// other boundary.  cnt_word is the single id -> word map; every access
// (OpCtx helpers and the raw producer/K0 sites) goes through it.
constexpr int kCntLine = 32;
__host__ __device__ constexpr int cnt_word(int id) { return id * kCntLine; }
static_assert(kCntCount * kCntLine * 4 == int(arena::kCountersPaddedBytes), "counter block size");

// Per-epoch arrival counts (must match the arrivals in the driver / op bodies).
// Boundary words: the arrival group is a fixed CTA set, independent of which
// CTAs have work (rank7 pad heads, CTAs 82..127 in QKVA, ...).
constexpr uint32_t kArriveK0Copy = 1;  // one publisher CTA (bookkeeping word, no waiter)
constexpr uint32_t kArriveIdsRx = 1;
constexpr uint32_t kArriveC0 = MEGA_ATTN_LEGACY_GEMV ? kGroup1Ctas : kQkvaCtaBlocks;
constexpr uint32_t kArriveC1 = kGroup1Ctas;
constexpr uint32_t kArriveC2 = kQlen + 1;  // four RoPE rows plus one local ids-acquire CTA
constexpr uint32_t kArriveC3 = kKvNormCtas;
constexpr uint32_t kArriveC4 = MEGA_ATTN_LEGACY_GEMV ? kK2Ctas : kHeads * kK2FusedProjqPerHead;
constexpr uint32_t kArriveC4Pe = kHeads * (kK2FusedCohort - kK2FusedProjqPerHead);  // 40 PE cohorts
constexpr uint32_t kArriveC6 = kK3SplitCtas;  // retired; kept as protocol-layout documentation
constexpr uint32_t kArriveC7 = kK4Ctas;
constexpr uint32_t kArriveK3Partials = kK3OptSplits;
constexpr uint32_t kArriveK3Stats = kK3OptSplits;
constexpr uint32_t kArriveK2Head = kK2FusedCohort;
constexpr uint32_t kArriveK3Head = kQlen * 2;  // two 256-dimension combine halves per query
constexpr uint32_t kArriveHeadCycle = kArriveK2Head + kArriveK3Head;
constexpr uint32_t kArriveIdsCopy =
    kIdsCopyCtas;  //  (K5Done / K5Hidden / K5Collective all retired: K5 arrives no word)
static_assert(kArriveHeadCycle == 20, "per-head K2 + K3 epoch cycle");

enum TraceId : int {
  kTrEntry = 0,     // kernel entry (before ring init)
  kTrPrologue = 1,  // after the single __syncthreads
  kTrK1Begin = 2,
  kTrK1End = 3,      // K1 body (Group1; idle CTAs 82..127: End == Begin)
  kTrFirstPage = 4,  // first ring page acquired by the consumers (item 0)
  kTrC0Arrive = 5,
  kTrC0WaitBegin = 6,
  kTrC0WaitSeen = 7,
  kTrC0WaitEnd = 8,
  kTrK2Begin = 9,
  kTrK2End = 10,
  kTrK2HeadArrive = 11,
  kTrK2HeadWaitBegin = 12,
  kTrK2HeadWaitSeen = 13,
  kTrK2HeadWaitEnd = 14,
  kTrC4Arrive = 15,
  kTrTailBegin = 16,
  kTrTailEnd = 17,  // tail CTA op body (kv-norm / RoPE / ids consume / idle)
  kTrPubWaitBegin = 18,
  kTrPubWaitEnd = 19,  // ids consume: kFlagPub sys-scope wait
  kTrC23Arrive = 20,   // tail arrives C3 (kv-norm) or C2 (RoPE group)
  kTrAttnWaitBegin = 21,
  kTrAttnWaitSeen = 22,
  kTrAttnWaitEnd = 23,  // C4 + C2 + C3
  kTrK3Begin = 24,
  kTrK3KvReady = 25,
  kTrK3PartialArrive = 26,
  kTrK3PartWaitBegin = 27,
  kTrK3PartWaitSeen = 28,
  kTrK3PartWaitEnd = 29,
  kTrK3End = 30,
  kTrC6Arrive = 31,
  kTrC6WaitBegin = 32,
  kTrC6WaitSeen = 33,
  kTrC6WaitEnd = 34,
  kTrK4Begin = 35,
  kTrK4End = 36,
  kTrC7Arrive = 37,
  kTrC7WaitBegin = 38,
  kTrC7WaitSeen = 39,
  kTrC7WaitEnd = 40,
  kTrK5Begin = 41,
  kTrK5UnprojEnd = 42,  // UnProj GEMV incl. local_hidden store
  kTrK5PushEnd = 43,    // the 6 x 32 inbox records issued (st.relaxed.sys.b128)
  kTrK5RxWaitBegin = 44,
  kTrK5RxWaitSeen = 45,
  kTrK5RxWaitEnd = 46,  // own-inbox poll (Seen = thread 0's record)
  // 47, 48: free (leader / K5Collective stamps)
  kTrK5ReduceEnd = 49,  // rank-ordered sum + residual + endpoint stores (after sync)

  kTrExit = 53,       // consumer thread 0 leaves the kernel (tails stamp it right after C2/C3)
  kTrProdBegin = 54,  // producer thread 0 starts streaming (first page issued next)
  kTrProdK3WaitBegin = 55,
  kTrProdK3WaitEnd = 56,  // producer: C2 + C3 poll before the KV gather
  kTrProdExit = 57,
  // rank 0 only (K0 publisher CTA 0; TP8 phase trace; the stamps are injected by
  kTrK0CopyEnd = 58,   // the ids / lengths copy of the publisher CTA is done (before its barrier)
  kTrK0PubStore = 59,  // st.release.sys kFlagPub issued
  kTrK0DoneWaitBegin = 60,  // the ack wait starts (+: kFlagAck, not kFlagDone)
  kTrK0DoneFirstSeen = 61,
  kTrK0DoneLastSeen = 62,  // min / max time at which the peers' kFlagAck were observed
  kTrEpoch = 63,           // args.epoch (slot validation)
};
static_assert(kTrEpoch < arena::kTraceStamps, "trace ids fit the per-CTA slot");

// ---------------------------------------------------------------------------

// layout  is kept for the attention tiles; activations that

// and K3 never overlap in time inside one CTA.
// ---------------------------------------------------------------------------
namespace smem {
constexpr int kGemmPartial = 0;  // f32 cross-warp partials: page-ring GEMV [8 warps|12 units][4
                                 // tokens][<=48 rows] <= 6144 B;
constexpr int kGemmPartialBytes = 48 * kQlen * 16 * 4;
constexpr int kActivation =
    kGemmPartialBytes;  // bf16 activations staged for the smem-operand GEMV (B operand):
                        //   K1 xnorm [4][6152] | K2 qnorm [4][2056] then post_q_nope_down head
                        //   slice [4][200] K4 post_o head slice [4][520] | K5 post_proj_o [4][2568]
// Row strides are padded by 8 bf16 (16 B) so the four token rows of one B fragment fall into
// different bank groups.
constexpr int kActPad = 8;
constexpr int kXnormStride = kHidden + kActPad;             // 6152 elements
constexpr int kQnormStride = kQlora + kActPad;              // 2056
constexpr int kProjqActStride = kNope + kActPad;            // 200
constexpr int kProjoActStride = kKvlora + kActPad;          // 520
constexpr int kUnprojActStride = kHeads * kVdim + kActPad;  // 2568
constexpr int kXnormBytes = kQlen * kXnormStride * 2;       // 49216 (largest staged activation)
constexpr int kQnormBytes = kQlen * kQnormStride * 2;
constexpr int kAttnQuery = kGemmPartialBytes;                      // bf16[10][584] = 11680
constexpr int kAttnKv = kAttnQuery + kHeads * kAttnStride * 2;     // bf16[36][584] = 42048
constexpr int kAttnIds = kAttnKv + kScoreSlots * kAttnStride * 2;  // i32[36] = 144
constexpr int kReduce = kAttnIds + kScoreSlots * 4;                // f32[32]
constexpr int kCorrCount = kReduce + 32 * 4;                       // i32
constexpr int kCorrIds = kCorrCount + 16;                          // i32[64]
#if MEGA_ATTN_LEGACY_GEMV
constexpr int kOpScratchBytes = ((kCorrIds + kQlen * 16 * 4) + 127) & ~127;
#else

// RMS reductions reuse the otherwise-dead GEMM-partial prefix, leaving K1's
// staged xnorm as the high-water mark.
constexpr int kOpScratchBytes = ((kActivation + kXnormBytes) + 127) & ~127;
#endif
static_assert(kActivation + kXnormBytes <= kReduce,
              "staged activation (xnorm) overflows into reduce scratch");
static_assert(kActivation + kQlen * kUnprojActStride * 2 <= kReduce,
              "staged post_proj_o overflows into reduce scratch");
static_assert(kGemmPartialBytes >= kSourceBlock * kQlen * 4 + kQlen * 32 * 4,
              "tile4 scratch (384*4 + results) must fit the gemm partial region");
static_assert(kOpScratchBytes % 128 == 0, "scratch alignment");
// Page ring: four exact-fit slots.  A split-KV half tile is the largest item:
// 32 rows x (512 bf16 values + 8 bf16 padding) = 0x8200 B.  Weight pages are
// at most 0x8080 B.  The tighter stride moves the persistent launch below the
// SM100 196-KiB shared-memory carveout boundary and leaves more unified L1.
// Handshake: full[4] mbarriers (arrival count = kProducerWarps, plus tx bytes)
// and empty[4] mbarriers (arrival count = kOpThreads) in static smem of the
// kernel; item i of a CTA uses slot i & 3 with parity (i >> 2) & 1 (consumers)
// / that parity ^ 1 (producers, first round passes).  One monotonic item
// counter per role spans all ops of the CTA (OpCtx::item, producer local).
constexpr int kRingSlots = 4;
constexpr int kRingSlotBytes = 0x8200;                   // 33280
constexpr int kRingBytes = kRingSlots * kRingSlotBytes;  // 133120
constexpr int kRingBarriers = 2 * kRingSlots;  // full[4] then empty[4], 8 B each (static smem)
constexpr int kPersistentDynamicBytes =
    kOpScratchBytes + kRingBytes;  // 194688 in the default build
constexpr int kStandaloneDynamicBytes =
    kOpScratchBytes + kRingBytes;  // wrappers carry the same ring (producer warps 8..11)
static_assert(kPersistentDynamicBytes + kRingBarriers * 8 <= 232448,
              "exceeds sm_100 max smem per block (227 KB)");
static_assert(kRingSlotBytes >= kPageBytes8080 && kRingSlotBytes % 128 == 0,
              "ring slot must hold the largest page, 128 B aligned");
static_assert(kK3OptHalfRows * kK3OptRingRowStride * 2 <= kRingSlotBytes,
              "K3 padded 32-row KV half tile must fit one ring slot");
static_assert(kPageBytes8080 % 16 == 0 && kUnprojPageBytes % 16 == 0 && kProjqPageBytes % 16 == 0 &&
                  kProjoPageBytes % 16 == 0,
              "16 B bulk copies");
static_assert(kGemmPartialBytes >= kOpWarps * kQlen * kUnprojRowsPerCta * 4 &&
                  kGemmPartialBytes >= 12 * kQlen * 16 * 4,
              "cross-warp partial buffer");
}  // namespace smem

#ifdef __CUDACC__
// ---------------------------------------------------------------------------
// Device-side helpers
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T* wsp(uint8_t* base, size_t offset) {
  return reinterpret_cast<T*>(base + offset);
}

__device__ __forceinline__ void store_release_sys(uint32_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}
__device__ __forceinline__ uint32_t load_acquire_sys(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void wait_epoch(const uint32_t* address, uint32_t epoch) {
  while (static_cast<int32_t>(load_acquire_sys(address) - epoch) < 0) {
  }
}
__device__ __forceinline__ uint32_t load_acquire_gpu(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}
__device__ __forceinline__ uint32_t load_relaxed_gpu(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}
__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
// Acquire-only fence: the PTX acquire pattern after a relaxed poll has observed the
// released value (used by every boundary / counter wait since; no release half).
__device__ __forceinline__ void fence_acquire_gpu() {
  asm volatile("fence.acquire.gpu;" ::: "memory");
}
// Fire-and-forget release increment.
__device__ __forceinline__ void red_release_gpu_add(uint32_t* address, uint32_t value) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}
// Used only after cp.async.bulk.wait_group has completed every payload byte.
// At that point the arrival word carries no additional release work.
__device__ __forceinline__ void red_relaxed_gpu_add(uint32_t* address, uint32_t value) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}
// Push collective: one 128-bit strong system-scope store / load per record.  PTX guarantees
// single-copy atomicity for the scalar .b128 access, so a record's payload and its epoch word are
// observed together. Word order: w0 = lower address.
__device__ __forceinline__ void st_relaxed_sys_b128(void* address, uint32_t w0, uint32_t w1,
                                                    uint32_t w2, uint32_t w3) {
  const uint64_t lo = uint64_t(w0) | (uint64_t(w1) << 32);
  const uint64_t hi = uint64_t(w2) | (uint64_t(w3) << 32);
  asm volatile(
      "{\n\t.reg .b128 r;\n\tmov.b128 r, {%1, %2};\n\tst.relaxed.sys.global.b128 [%0], r;\n\t}" ::
          "l"(address),
      "l"(lo), "l"(hi)
      : "memory");
}
__device__ __forceinline__ void ld_relaxed_sys_b128(const void* address, uint64_t& lo,
                                                    uint64_t& hi) {
  asm volatile(
      "{\n\t.reg .b128 r;\n\tld.relaxed.sys.global.b128 r, [%2];\n\tmov.b128 {%0, %1}, r;\n\t}"
      : "=l"(lo), "=l"(hi)
      : "l"(address)
      : "memory");
}
// Counting arrive with the last-arriver role (K5Done): one acq_rel atom, no separate fences.
__device__ __forceinline__ uint32_t atom_acq_rel_gpu_add(uint32_t* address, uint32_t value) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(address), "r"(value)
               : "memory");
  return old;
}
__device__ __forceinline__ void spin_backoff() {
#if MEGA_ATTN_WAIT_SLEEP_NS > 0
  __nanosleep(MEGA_ATTN_WAIT_SLEEP_NS);
#endif
}

// Monotonic epoch-based arrival counter (internal K0/K3/K5 counters).  Never
// reset: the arena is zeroed once at init, every epoch adds exactly `count`
// arrivals, so the wait target for epoch e is count*e.  signal = __threadfence +
// atomicAdd (the return value identifies the last arriver); wait = acquire spin.
struct ArrivalCounter {
  uint32_t* word;
  __device__ __forceinline__ static uint32_t target(uint32_t count, uint32_t epoch) {
    return count * epoch;
  }
  // Returns the post-increment value; equal to target(count, epoch) for the
  // last arriver of this epoch.  Fence before publishes this CTA's prior global
  // writes; fence after lets the last arriver read everyone else's.
  __device__ __forceinline__ uint32_t arrive() {
    __threadfence();
    uint32_t seen = atomicAdd(word, 1u) + 1u;
    __threadfence();
    return seen;
  }
  // relaxed poll + one acquire-only fence (the boundary idiom) instead of an acquire load per poll.
  __device__ __forceinline__ void wait(uint32_t target_value) const {
    while (static_cast<int32_t>(load_relaxed_gpu(word) - target_value) < 0) {
      spin_backoff();
    }
    fence_acquire_gpu();
  }
};

__device__ __forceinline__ bool word_reached(const uint32_t* word, uint32_t target_value) {
  return static_cast<int32_t>(load_relaxed_gpu(word) - target_value) >= 0;
}

__device__ __forceinline__ int k3_rows_per_split(int /*length*/) { return kK3OptRows; }

// Balanced split geometry: the query's live selection is cut into 16-row
// granules distributed evenly over the 32 split CTAs, heavier counts first,
// so every split's key range is contiguous and ascending in split order (the
// fixed ascending order of the global normalization and the combine is
// preserved) and populated splits form a prefix.  A full 2048-key selection
// reproduces the original 64-key splits exactly (so the >2048-visible LL id
// rows keep their original 64-id-per-CTA ownership); shorter live selections
// shrink every split's gather, QK and PV instead of idling the tail splits.
// The per-split local maxima move with the geometry; every statistic,
// exponential, normalization and combine term remains FP32 with the same
// global formula, inside the frozen output gate.
__device__ __forceinline__ int k3_split_granules(int length) { return (length + 15) >> 4; }
__device__ __forceinline__ int k3_split_row_lo(int length, int split) {
  const int granules = k3_split_granules(length);
  const int share = granules >> 5;
  const int rem = granules & (kK3OptSplits - 1);
  return (split * share + min(split, rem)) << 4;
}
__device__ __forceinline__ int k3_split_row_cnt(int length, int split) {
  const int granules = k3_split_granules(length);
  const int share = granules >> 5;
  const int rem = granules & (kK3OptSplits - 1);
  return min((share + (split < rem ? 1 : 0)) << 4, kK3OptRows);
}
__device__ __forceinline__ int k3_live_splits(int length) {
  return min(k3_split_granules(length), kK3OptSplits);
}

// ---- mbarrier / bulk-copy primitives (PTX ISA 8.x, sm_90+) -------------------------
// Shared-memory addresses are 32-bit shared-window offsets (cvta), as the
// mbarrier and cp.async.bulk instructions require.
__device__ __forceinline__ uint32_t smem_u32(const void* generic) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(generic));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t arrival_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(arrival_count) : "memory");
}
// Make the inits visible to the async proxy (TMA complete_tx) and other threads.
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
// Spin until the phase with the given parity has completed.
__device__ __forceinline__ void mbar_wait_parity(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "MBAR_WAIT_LOOP:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra MBAR_WAIT_DONE;\n\t"
      "bra MBAR_WAIT_LOOP;\n\t"
      "MBAR_WAIT_DONE:\n\t"
      "}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_count(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx_bytes)
               : "memory");
}
// One bulk copy global -> shared, completion counted on `bar`.
__device__ __forceinline__ void bulk_copy_g2s(uint32_t dst_smem, const void* src_global,
                                              uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(dst_smem),
      "l"(src_global), "r"(bytes), "r"(bar)
      : "memory");
}

// Per-op context: which CTA of the op's CTA set this is, thread role, the smem
// carve-out and the page-ring handshake.  Op bodies run on the consumer warps
// only (threads 0..kOpThreads-1) and synchronize with a named barrier so
// producer warps can run concurrently in the same CTA.
struct OpCtx {
  int op;
  int cta;       // index within the op's CTA set
  int num_ctas;  // size of the op's CTA set
  int tid;       // == threadIdx.x for consumers (0..255)
  int warp;      // consumer warp 0..7
  int lane;
  bool producer;       // warp role (true for warps 8..11)
  uint8_t* smem;       // op scratch base (smem::kOpScratchBytes)
  uint8_t* ring;       // page ring base (smem::kRingBytes), fed by the producer warps
  uint8_t* arena;      // local arena (args.peers[args.rank])
  uint32_t* counters;  // arena + arena::kCounters
  uint32_t ring_s;     // shared-window address of ring slot 0
  uint32_t full_s;     // shared-window address of full[0]  (full[s] = full_s + 8 s)
  uint32_t empty_s;    // shared-window address of empty[0]
  uint32_t item;       // next page item of this CTA (monotonic across ops; == producer order)
  unsigned long long* trace;

  __device__ __forceinline__ void sync() const {
    asm volatile("bar.sync 1, %0;" ::"n"(kOpThreads) : "memory");
  }
  __device__ __forceinline__ ArrivalCounter counter(int id) const {
    return ArrivalCounter{counters + cnt_word(id)};
  }
  // Boundary words (driver only; CTA-uniform calls by all 256 consumer threads).
  __device__ __forceinline__ void boundary_arrive(int id) const {
    sync();
    if (tid == 0) red_release_gpu_add(counters + cnt_word(id), 1u);
  }
  // Use when the op body has already ended with a consumer barrier.  That
  // barrier orders every producer thread's global stores before lane 0's
  // release reduction and avoids a redundant second 256-thread rendezvous.
  __device__ __forceinline__ void boundary_arrive_presynced(int id) const {
    if (tid == 0) red_release_gpu_add(counters + cnt_word(id), 1u);
  }
  // `tr`: TraceId of the wait's Begin stamp; Seen = tr + 1, End = tr + 2.
  __device__ __forceinline__ void boundary_wait(int id, uint32_t target_value, int tr = -1) const {
    if (tid == 0) {
      if (tid == 0) (void)0;
      while (!word_reached(counters + cnt_word(id), target_value)) spin_backoff();
      if (tid == 0) (void)0;
      fence_acquire_gpu();
    }
    sync();
    if (tid == 0) (void)0;
  }
  __device__ __forceinline__ void boundary_wait2(int id0, uint32_t t0, int id1, uint32_t t1,
                                                 int tr = -1) const {
    if (tid == 0) {
      if (tid == 0) (void)0;
      for (;;) {
        const bool r0 = word_reached(counters + cnt_word(id0), t0);
        const bool r1 = word_reached(counters + cnt_word(id1), t1);
        if (r0 && r1) break;
        spin_backoff();
      }
      if (tid == 0) (void)0;
      fence_acquire_gpu();
    }
    sync();
    if (tid == 0) (void)0;
  }
  __device__ __forceinline__ void boundary_wait3(int id0, uint32_t t0, int id1, uint32_t t1,
                                                 int id2, uint32_t t2, int tr = -1) const {
    if (tid == 0) {
      if (tid == 0) (void)0;
      for (;;) {
        const bool r0 = word_reached(counters + cnt_word(id0), t0);
        const bool r1 = word_reached(counters + cnt_word(id1), t1);
        const bool r2 = word_reached(counters + cnt_word(id2), t2);
        if (r0 && r1 && r2) break;
        spin_backoff();
      }
      if (tid == 0) (void)0;
      fence_acquire_gpu();
    }
    sync();
    if (tid == 0) (void)0;
  }
  template <typename T>
  __device__ __forceinline__ T* scratch(int offset) const {
    return reinterpret_cast<T*>(smem + offset);
  }
  // Page ring, consumer side (all 256 consumer threads call both, in page order).
  // acquire_page: wait until the producer's bulk copy of the current item has
  // landed (full[slot], parity (item>>2)&1); returns the slot's shared address.
  __device__ __forceinline__ uint32_t acquire_page() const {
    const uint32_t slot = item & (smem::kRingSlots - 1);
    mbar_wait_parity(full_s + slot * 8, (item >> 2) & 1u);
    if (item == 0 && tid == 0) (void)0;
    return ring_s + slot * smem::kRingSlotBytes;
  }
  // release_page: each warp first orders every lane's final shared load, then its
  // leader retires all 32 lanes with one counted arrival.  Eight warp leaders
  // still decrement the empty barrier's 256-thread count exactly, while
  // avoiding 256 independent mbarrier arrivals per streamed page.
  __device__ __forceinline__ void release_page() {
    const uint32_t slot = item & (smem::kRingSlots - 1);
    __syncwarp();
    if (lane == 0) mbar_arrive_count(empty_s + slot * 8, 32);
    ++item;
  }
};

// Ring barrier init: lanes in the second producer warp initialize full[t]
// (count = producer warps) and empty[t] (count = consumer threads); the caller
// follows with the kernel's single __syncthreads.  Keeping this off producer
// warp 0 lets the initialization overlap both the K1 consumer entry loads and
// the slot-0 preissue.  `first_slot` leading slots were initialized earlier.
// `bars` = static smem uint64_t[smem::kRingBarriers].
__device__ __forceinline__ void ring_init_barriers(uint64_t* bars, int first_slot = 0) {
  const int init_lane = static_cast<int>(threadIdx.x) - (kOpThreads + 32);
  const int slot = init_lane + first_slot;
  if (init_lane >= 0 && slot < smem::kRingSlots) {
    mbar_init(smem_u32(bars + slot), kProducerWarps);
    mbar_init(smem_u32(bars + smem::kRingSlots + slot), kOpThreads);
    mbar_fence_init();
  }
}

__device__ __forceinline__ OpCtx make_op_ctx(int op, int cta, int num_ctas, uint8_t* smem,
                                             uint8_t* ring, uint64_t* ring_bars, uint8_t* arena) {
  OpCtx ctx;
  ctx.op = op;
  ctx.cta = cta;
  ctx.num_ctas = num_ctas;
  ctx.tid = static_cast<int>(threadIdx.x);
  ctx.warp = ctx.tid >> 5;
  ctx.lane = ctx.tid & 31;
  ctx.producer = ctx.warp >= kOpWarps;
  ctx.smem = smem;
  ctx.ring = ring;
  ctx.arena = arena;
  ctx.counters = reinterpret_cast<uint32_t*>(arena + arena::kCountersPadded);
  ctx.ring_s = smem_u32(ring);
  ctx.full_s = smem_u32(ring_bars);
  ctx.empty_s = smem_u32(ring_bars + smem::kRingSlots);
  ctx.item = 0;
  ctx.trace = nullptr;
  return ctx;
}

// Op bodies (defined in kernel.cu).  Consumer warps only.  `static`: the
// bodies live in headers included by both persistent_kernel.cu and
// standalone_wrappers.cu; nvcc emits host shadow symbols for external-linkage
// __device__ functions, so internal linkage is required to link the two TUs.
static __device__ void op_k0_rank0_publish(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k1_qkva_cache(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k1_tails(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k2_wqb(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k2_projq(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k2_fused(const MegaArgs& args, OpCtx& ctx);  //  (page-ring build only)
static __device__ void op_k3_attention(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k4_combine_projo(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k5_unproj_collective(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k5_unproj_phase(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k5_collective_phase(const MegaArgs& args, OpCtx& ctx);
static __device__ void op_k5_unproj_only(const MegaArgs& args, OpCtx& ctx);

// Producer page lists: entry `index` of the weight pages CTA `cta` of op `op`
// consumes; returns false past the end (also for CTAs outside the op's page
// set).  bytes is a multiple of 16; bytes == 0 marks a skipped page (rank7 pad
// heads) that is NOT a ring item on either side.
static __device__ bool op_page(int op, const MegaArgs& args, int cta, int index,
                               const uint8_t** ptr, int* bytes);
#endif  // __CUDACC__

}  // namespace flashinfer::glm5_tp8::mla_q4
