/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#if defined(INDEXER_IDS_WIRE_NATIVE) && INDEXER_IDS_WIRE_NATIVE != 1
#error "The production attention interface requires the LL selected-ID wire"
#endif
#ifndef INDEXER_IDS_WIRE_NATIVE
#define INDEXER_IDS_WIRE_NATIVE 1
#endif
#if defined(INDEXER_COLLECTIVE_WIRE_NATIVE) && INDEXER_COLLECTIVE_WIRE_NATIVE != 0
#error "The production attention interface requires the self-tagged TP8 collective"
#endif
#ifndef INDEXER_COLLECTIVE_WIRE_NATIVE
#define INDEXER_COLLECTIVE_WIRE_NATIVE 0
#endif
#define INDEXER_PEERS_INBOX_BASES (INDEXER_IDS_WIRE_NATIVE && INDEXER_COLLECTIVE_WIRE_NATIVE)

#ifdef __CUDACC__
#define INDEXER_HD __host__ __device__
#else
#define INDEXER_HD
#endif

namespace flashinfer::glm5_tp8::indexer_q4 {

// ---------------------------------------------------------------------------
// 1. Grid, thread roles, model constants
// ---------------------------------------------------------------------------
constexpr int kWorld = 8;
constexpr int kGrid = 148;   // 148 CTAs = SMs on B200 (WarpsGroup<256,0,148,0>)
constexpr int kBlock = 384;  // 256 consumer + 128 prefetcher threads
constexpr int kConsumerWarps = 8;
constexpr int kConsumerThreads = kConsumerWarps * 32;  // 256 (tid < 0x100)
constexpr int kPrefetchWarps = 4;
constexpr int kPrefetchThreads = kPrefetchWarps * 32;
constexpr int kConsumerMaxReg = 216;
constexpr int kPrefetchMaxReg = 72;
static_assert(kConsumerThreads * kConsumerMaxReg + kPrefetchThreads * kPrefetchMaxReg ==
                  168 * kBlock,
              "register split must equal the 168 x 384 register budget");

constexpr int kSeqLen = 4;                   // S = kSeqLen (4 query tokens / launch)
constexpr int kHidden = 6144;                // x row (bf16)
constexpr int kQaDim = 2048;                 // q_a = indexer q_lora_rank (RMSNorm 1/2048)
constexpr int kIdxDim = 128;                 // indexer head dim (ki / iq_rt rows are 128 bf16)
constexpr int kIdxHeads = 32;                // N heads (ctx row 3 +0x18; D: 32)
constexpr int kIqDim = kIdxHeads * kIdxDim;  // 4096 = IQ row (token stride 0x2000 B)
constexpr float kHadamardScale =
    0.08838834613561630249f;          // 128^-0.5 = 0x3db504f3: the scale after H_128 (P3, Group2a)
constexpr int kKiRows = 128;          // rows of W_ki (CTAs 64..67 of P1)
constexpr int kTopk = 2048;           // k = 0x800
constexpr int kMaxSeqLen = 202760;    // ki_cache rows = max_seq_len 202752 + kv_cache_pad 8
constexpr int kDenseCurPos = 0x7fc;   // cur_pos >= 0x7fc: scoring / K pages / Group2b
constexpr int kSelectCurPos = 0x7fd;  // cur_pos >= 0x7fd: selection stage B/C
constexpr int kLastDenseCurPos = 0x7ff;  // cur_pos <= 0x7ff: at least one dense q
INDEXER_HD constexpr bool is_sparse(int cur_pos) { return cur_pos >= kDenseCurPos; }
INDEXER_HD constexpr int total_len(int cur_pos) { return cur_pos + kSeqLen; }  // L = cur_pos + 4

// ---------------------------------------------------------------------------
// 2. CTA groups and phase CTA sets
// ---------------------------------------------------------------------------
constexpr int kGroup1Ctas = 128;
constexpr int kGroup2aBase = 128;  // 128..131 (ctaid > 0x83 -> Group2b)
constexpr int kGroup2aCtas = 4;
constexpr int kGroup2aWorker = 128;
constexpr int kGroup2bBase = 132;  // 132..147 (ctaid > 0x93 -> assert)
constexpr int kGroup2bCtas = 16;
constexpr int kGroup2bWorkCtas = kSeqLen;  // 132..135 refine query q = ctaid - 132; 136..147 idle
static_assert(kGroup2aBase + kGroup2aCtas == kGroup2bBase && kGroup2bBase + kGroup2bCtas == kGrid,
              "CTA groups");

constexpr int kP1Ctas = 68;   // wqakis GEMV: ctaid <= 0x43 (0..63 q_a, 64..67 ki)
constexpr int kP1bBase = 68;  // W_is bf16 GEMV: 68..71 (ctaid > 0x47 -> idle)
constexpr int kP1bCtas = 4;
constexpr int kP1AllCtas = kP1Ctas + kP1bCtas;  // 72; CTAs 72..127 idle in P1
constexpr int kP2Ctas = kGroup1Ctas;            // Wqi GEMV: all 128
INDEXER_HD constexpr int p3_head_blocks(int n_heads) { return (n_heads + 15) / 16; }  // ceil(N/16)
INDEXER_HD constexpr int p3_ctas(int n_heads) {
  return kSeqLen * p3_head_blocks(n_heads);
}  // 4*ceil(N/16)
// P3 CTA c -> query q = c / ceil(N/16), head block hb = c % ceil(N/16) (signed division);
// row = 16*hb + (tid >> 4), lane = tid & 15; rows >= N skip
INDEXER_HD constexpr int p3_query_of_cta(int cta, int n_heads) {
  return cta / p3_head_blocks(n_heads);
}
INDEXER_HD constexpr int p3_head_block_of_cta(int cta, int n_heads) {
  return cta % p3_head_blocks(n_heads);
}
constexpr int kP3RowsPerBlock = 16;  // (q, head) rows per P3 CTA = 256 threads / 16 lanes
constexpr int kP3CtasDefault = p3_ctas(kIdxHeads);  // 8
constexpr int kP4Ctas = kGroup1Ctas;                // scoring: tiles for ctaid*128 < L
constexpr int kP5Ctas = kGroup1Ctas;                // selection stage B: ctaid < N_active
constexpr int kP6Ctas = 8;                          // ids broadcast: ctaid < 8
constexpr int kP7Ctas = kGroup1Ctas;                // collective: all 128 (48 columns each)

// ---------------------------------------------------------------------------
// 3. Weight page geometry
// ---------------------------------------------------------------------------
constexpr int kPageBytes8080 =
    0x8080;  // W8A16 page: 32 rows x 1024 K fp8 + 8 f32 scales @+0x8000 + 96 B zero
constexpr int kPageMatBytes = 0x8000;
constexpr int kPageScaleOffset = 0x8000;  // scale j (0..7) = 128-K block j, indexed by warp
constexpr int kPageScales = 8;
constexpr int kWqakisCtas = kP1Ctas;  // 68 CTAs x 6 pages
constexpr int kWqakisPages = 6;
constexpr int kWqakisCtaStride = kWqakisPages * kPageBytes8080;          // 0x30300
constexpr size_t kWqakisBytes = size_t(kWqakisCtas) * kWqakisCtaStride;  // 13,421,568 = 0xCCCC00
constexpr int kWisRowBlocks = 2;  // bf16 MMA pages: 2 row blocks x 6 pages x 0x8000
constexpr int kWisPages = 6;
constexpr int kWisPageBytes = 0x8000;                                     // no scales
constexpr int kWisRowBlockStride = kWisPages * kWisPageBytes;             // 0x30000
constexpr size_t kWisBytes = size_t(kWisRowBlocks) * kWisRowBlockStride;  // 393,216
constexpr int kWqiCtas = kGroup1Ctas;                                     // 128 CTAs x 2 pages
constexpr int kWqiPages = 2;
constexpr int kWqiCtaStride = kWqiPages * kPageBytes8080;       // 0x10100
constexpr size_t kWqiBytes = size_t(kWqiCtas) * kWqiCtaStride;  // 8,421,376 = 0x808000
static_assert(kWqakisBytes == 13421568 && kWisBytes == 393216 && kWqiBytes == 8421376,
              "weight blob sizes (D 3.1)");
// P1b CTA -> (row block, token pair): rowblock = (c-68) >= 2 ? c-70 : c-68; tokens (c-68) >= 2 ?
// {2,3} : {0,1}
INDEXER_HD constexpr int wis_rowblock(int cta) {
  return (cta - kP1bBase) >= 2 ? cta - kP1bBase - 2 : cta - kP1bBase;
}
INDEXER_HD constexpr int wis_token_base(int cta) { return (cta - kP1bBase) >= 2 ? 2 : 0; }

// ---------------------------------------------------------------------------
// 4. Page ring  and K tiles
// ---------------------------------------------------------------------------
constexpr int kRingSlots = 4;                            // item k uses slot k & 3
constexpr int kRingSlotBytes = 0x9000;                   // kPageSize = 36864
constexpr int kRingBytes = kRingSlots * kRingSlotBytes;  // 0x24000
constexpr uint32_t kFullArrivals =
    4;  // full[s]: 4 prefetch-warp lane-0 arrivals + tx bytes (encoding 0x100000-4)
constexpr uint32_t kEmptyArrivals = 256;  // empty[s]: 256 consumer arrivals
constexpr int kKTileRows = 128;           // one K tile = 128 kv rows x 128 dim bf16
constexpr int kKTileBoxBytes =
    0x4000;  // TMA box: 64 dim x 128 rows bf16 (expect_tx 0x8000 for 2 boxes)
constexpr int kKTileBytes = 2 * kKTileBoxBytes;         // 0x8000 <= slot
constexpr int kKTileStrideRows = kP4Ctas * kKTileRows;  // 16384: CTA c owns tiles c, c+128...
constexpr int kQTileBoxRows =
    32;  // q tile TMA box = 32 rows x 64 dims bf16 = 4 KB (0x1000 B destination stride)
constexpr int kQTileBytes = 0x8000;  // 8 boxes; the q-tile mbarrier's tx count
constexpr uint64_t kL2PolicyEvictNormal =
    0x1000000000000000ull;  // L2 cache hint of the K-tile / q-tile TMA loads and weight pages
// n_tiles(c) = ctaid*128 < L ? ceil((L - ctaid*128) / 16384) : 0
INDEXER_HD constexpr int k_tiles_of_cta(int cur_pos, int cta) {
  return (cta * kKTileRows < total_len(cur_pos))
             ? (total_len(cur_pos) - cta * kKTileRows + kKTileStrideRows - 1) / kKTileStrideRows
             : 0;
}
// N_active = min(ceil(L/128), 128)
INDEXER_HD constexpr int n_active_ctas(int cur_pos) {
  const int n = (total_len(cur_pos) + kKTileRows - 1) / kKTileRows;
  return n > kGroup1Ctas ? kGroup1Ctas : n;
}

// ---------------------------------------------------------------------------
// 5. Scoring / selection constants
// ---------------------------------------------------------------------------
constexpr float kScoreScale = 0.015625f;  // 0.015625 = 1/64 = 128^-0.5 * 32^-0.5
constexpr int kHistBins = 256;            // fp16-key high byte
constexpr int kHistBlockDwords = 258;     // [256 suffix counts][n_sure][n_cand]
constexpr int kHistBlockBytes =
    kHistBlockDwords * 4;            // 0x408 (q stride of the H block and of the smem histogram)
constexpr int kHistSureIndex = 256;  // ticket word +0x400
constexpr int kHistCandIndex = 257;  // ticket word +0x404
constexpr int kRefineMaxCand = 16384;
constexpr int kRefineRounds = 3;  // 8-bit digits of the fp32 key, shift 16 - 8r
constexpr int kTmemColumns = 256;
// x 32 columns per bit, column base idx*0x20 -> 128 (arithmetic)
// tcgen05.mma operands of the scoring step: instruction descriptor below; smem descriptor hi = SBO
// 1024 B (8 rows x 128 B), version 1, layout 2 = SWIZZLE_128B; lo = (start address >> 4) | LBO
// field 1.  K16 steps advance lo by 2 (32 B) inside a 64-dim box and by 0x400 (0x4000 B) to the
// second box.  Operand roles: A = K tile (M = kv rows), B = q tile (N = 4 q x 32 heads).
constexpr uint32_t kUmmaIdescBf16F32M128N128 =
    0x08200490u;  // D f32 (bits 4-5 = 1), A/B bf16 (bits 7-9 / 10-12 = 1), K-major, N 128 (bits
                  // 17-22 = 16), M 128 (bits 24-28 = 8)
constexpr uint32_t kUmmaSmemDescHi = 0x40004040u;
constexpr uint32_t kUmmaSmemDescLboField = 1u << 16;
INDEXER_HD constexpr uint32_t umma_smem_desc_lo(uint32_t window_addr) {
  return ((window_addr >> 4) & 0x3fc0u) | kUmmaSmemDescLboField;
}
constexpr uint32_t kUmmaKStepLo = 2;        // + K16 step inside a 128 B swizzled row (32 B >> 4)
constexpr uint32_t kUmmaBoxStepLo = 0x400;  // + second 64-dim box (0x4000 B >> 4)
constexpr int kUmmaKStepsPerTile = 8;       // 128 dims / 16
constexpr int kLogitsStride =
    kMaxSeqLen;  // fixed logits row stride (pitch L = cur_pos + 4 under kFlagNativePitch;
                 //   see logits_row_stride)
constexpr int kWsStride =
    200 * 1024 + 260;  // fixed candidate row stride = IDX_SEL_WS per-query int32 (205060)
                       //   (see ws_row_stride)

// ---------------------------------------------------------------------------
// 6. Cross-GPU records
// ---------------------------------------------------------------------------
constexpr int kIdsRecordBytes =
    16;  // {id0, tag, id1, tag} -> peer_k + r*16, r = ctaid*256 + tid (+0x800 per round)
constexpr int kIdsPerRecord = 2;
constexpr int kRecvBufBytes = 8 * kTopk * 2 * 4;  // pure rank recv_buf int32[8][2048][2] = 0x20000
constexpr int kInboxBlockBytes =
    128;  // 8 lanes x 16 B: lines 0..5 = 48 bf16, lines 6,7 = {epoch x4}
constexpr int kInboxSlots = kWorld - 1;  // 7 source slots
constexpr int kInboxParities = 2;        // epoch parity double buffer
constexpr int kInboxCtas = kP7Ctas;      // 128
// peer[dst] + ((q*0x700 + ctaid) << 7) + parity*0x4000 + slot*0x8000 + line*16
//   == [q 4][slot 7][parity 2][cta 128][128 B]
constexpr size_t kInboxParityBytes = size_t(kInboxCtas) * kInboxBlockBytes;     // 0x4000
constexpr size_t kInboxSlotBytes = size_t(kInboxParities) * kInboxParityBytes;  // 0x8000
constexpr size_t kInboxQBytes = size_t(kInboxSlots) * kInboxSlotBytes;  // 0x38000 = 0x700 << 7
constexpr size_t kInboxBytes = size_t(kSeqLen) * kInboxQBytes;          // 0xE0000 = 896 KB
static_assert(kInboxQBytes == (size_t(0x700) << 7), "inbox q stride must match (q*0x700+ctaid)<<7");
INDEXER_HD constexpr size_t inbox_block_offset(int q, int slot, int parity, int cta) {
  return size_t(q) * kInboxQBytes + size_t(slot) * kInboxSlotBytes +
         size_t(parity) * kInboxParityBytes + size_t(cta) * kInboxBlockBytes;
}
constexpr int kCollectiveColsPerCta = kHidden / kP7Ctas;  // 48 bf16 per (CTA, q) = lines 0..5
constexpr uint32_t kP7GroupMaskBase = 0x11111111u;
INDEXER_HD constexpr uint32_t p7_group_mask(int tid) { return kP7GroupMaskBase << (tid & 3); }
// NaN sentinel fills of P7 (both wires): idx = ctaid*256 + tid <= 0x3ff -> 16 B of 0x7fc00000 at
// f32 target + idx*16, <= 0x47f -> 16 B of 0x7fc07fc0 at bf16 target + idx*16, only when BOTH ctx
// row 7 +0x20 / +0x28 are non-null; replace mode fills the same 1024 / 1152

constexpr int kNanFillF32Vec = 1024;
constexpr int kNanFillBf16Vec = 1152;
constexpr uint32_t kNanFillF32Word = 0x7fc00000u;
constexpr uint32_t kNanFillBf16Word = 0x7fc07fc0u;

// ---------------------------------------------------------------------------
// 6b. Our attention's wire (INDEXER_IDS_WIRE_NATIVE=0 publish / INDEXER_COLLECTIVE_WIRE_NATIVE=0
// reduce): the `namespace arena` of the attention kernel's mega_attn_abi.h mirrored VALUE FOR VALUE
// (line numbers quoted; every number is static_asserted against the literal below).  These offsets
// address the PURE RANKS' arenas (rank 0 writes peers[r] + kIdsRx / kFlagPub, polls peers[r] +
// kFlagAck) and rank 0's own arena (the pure ranks push their K5 slice records to peers[0] +
// k5_rx_offset(...)) -- byte compatibility here IS the protocol [COMM_PROTOCOL_COMPAT 0.2, 1-3].
// ---------------------------------------------------------------------------
namespace mega_wire {
constexpr size_t kFlagPub = 0;     //  rank 0 -> st.release.sys [peers[r] + kFlagPub] = epoch; pure
                                   //  rank r's tail CTA 136 waits on its LOCAL word
constexpr size_t kFlagAck = 4;     //  pure rank r -> its own word = epoch right after observing
                                   //  kFlagPub; rank 0 CTA 0 lanes 1..7 poll peers[r] + kFlagAck
constexpr size_t kStats = 590848;  //  (recorded only)
constexpr int kRxSlots = kWorld - 1;   //  7 source slots, slot r-1 = pure rank r
constexpr int kRxRecordElems = 6;      //  bf16 per record (12 B) + u32 epoch
constexpr int kRxRecordsPerToken = 8;  //  = 48 rows per K5 CTA / 6
constexpr size_t kRxRecordBytes =
    16;  //  {dword0..2 = bf16 pairs (low half = lower row), dword3 = epoch}
constexpr size_t kRxCtaBytes = size_t(kSeqLen) * kRxRecordsPerToken * kRxRecordBytes;  //  512
constexpr int kRxCtas =
    128;  // / kUnprojCtas == kK5Ctas: CTA c owns endpoint columns [48c, 48c + 48)
constexpr int kRowsPerCta = kHidden / kRxCtas;                      //  kUnprojRowsPerCta = 48
constexpr size_t kRxSlotBytes = size_t(kRxCtas) * kRxCtaBytes;      //  65536
constexpr size_t kRxParityBytes = size_t(kRxSlots) * kRxSlotBytes;  //  458752
constexpr size_t kHiddenRx =
    5309952;  //  [parity = epoch & 1][slot 0..6][cta 0..127][q 0..3][record 0..7]
constexpr size_t kHiddenRxBytes = size_t(2) * kRxParityBytes;  //  917504
constexpr size_t kExitCredit = 512;  //  [cta 0..127] 16 B {E,E,E,E}: rank 0 -> pure rank, last
                                     //  epoch whose kHiddenRx records rank 0 consumed
constexpr size_t kExitCreditLineBytes = 16;
constexpr size_t kTrace =
    kHiddenRx + kHiddenRxBytes;  //  6227456 (pure-rank phase trace; rank 0 never touches it)
constexpr size_t kTraceBytes = 1212416;  //  16 epochs x 148 CTAs x 64 u64
constexpr size_t kIdsRx =
    kTrace + kTraceBytes;  //  7439872 [parity]: ids int32[4][2048] ++ lengths int32[4] ++ pad
constexpr size_t kIdsRxLengths =
    size_t(kSeqLen) * kTopk * 4;  //  32768 = byte offset of lengths[4] inside a parity buffer
constexpr size_t kIdsRxParityBytes = kIdsRxLengths + 128;      //  32896
constexpr size_t kIdsRxBytes = size_t(2) * kIdsRxParityBytes;  //  65792
constexpr size_t kArenaBytes =
    size_t(8) * 1024 * 1024;  //  the pure ranks' allocation (mega_attn_arena_bytes)
// wire-layout literals (one assert each so a drift names itself)
static_assert(kFlagPub == 0, "wire kFlagPub");
static_assert(kFlagAck == 4, "wire kFlagAck");
static_assert(kHiddenRx == 5309952, "wire kHiddenRx");
static_assert(kRxParityBytes == 458752, "wire kRxParityBytes");
static_assert(kRxSlotBytes == 65536, "wire kRxSlotBytes");
static_assert(kRxCtaBytes == 512, "wire kRxCtaBytes");
static_assert(kIdsRx == 7439872, "wire kIdsRx");
static_assert(kExitCredit == 512 && kExitCreditLineBytes == 16, "wire kExitCredit");
static_assert(kIdsRxParityBytes == 32896, "wire kIdsRxParityBytes");
static_assert(kIdsRxLengths == 32768, "wire kIdsRxLengths");
static_assert(kHiddenRxBytes == 917504 && kTrace == 6227456 && kTraceBytes == 1212416 &&
                  kIdsRxBytes == 65792 && kArenaBytes == 8388608 && kStats == 590848,
              "wire derived constants");
static_assert(
    kRxRecordsPerToken * kRxRecordElems == kRowsPerCta &&
        kRxRecordsPerToken * kRxRecordBytes == 128 && kRowsPerCta * kRxCtas == kHidden,
    "one token's 48-element slice = 8 records = one 128 B line (mega_attn_abi.h:376-377)");
static_assert(kHiddenRx % 128 == 0 && kIdsRx % 128 == 0 && kHiddenRx + kHiddenRxBytes <= kTrace &&
                  kTrace + kTraceBytes <= kIdsRx && kIdsRx + kIdsRxBytes <= kArenaBytes,
              "wire layout consistency");
// Publish payload: ids int32[4][2048] immediately followed by lengths int32[4] = 32,784 B (16 B
// multiple), written as ONE bulk copy to peers[dst] + ids_rx_offset(epoch).
constexpr uint32_t kIdsPayloadBytes = uint32_t(kIdsRxLengths);                      // 32768
constexpr uint32_t kLengthsPayloadBytes = uint32_t(kSeqLen) * 4;                    // 16
constexpr uint32_t kPublishPayloadBytes = kIdsPayloadBytes + kLengthsPayloadBytes;  // 32784
static_assert(kPublishPayloadBytes % 16 == 0 && kPublishPayloadBytes <= kIdsRxParityBytes,
              "ids + lengths bulk payload fits a parity buffer");
// Record k (0..7) of token q of CTA `cta` in source slot `slot` of an inbox (k5_rx_offset).
INDEXER_HD constexpr size_t k5_rx_offset(uint32_t epoch, int slot, int cta, int q, int k) {
  return kHiddenRx + size_t(epoch & 1u) * kRxParityBytes + size_t(slot) * kRxSlotBytes +
         size_t(cta) * kRxCtaBytes + size_t(q) * kRxRecordsPerToken * kRxRecordBytes +
         size_t(k) * kRxRecordBytes;
}
// Parity buffer of the ids publish of `epoch`.
INDEXER_HD constexpr size_t ids_rx_offset(uint32_t epoch) {
  return kIdsRx + size_t(epoch & 1u) * kIdsRxParityBytes;
}
// Published length of query q: the pure rank uses ids[q][slot] only for slot < length and masks id
// <= cur_pos + q; the dense_ids rule min(kSelected, cur_pos + q + 1) is what we publish for every
// launch: sparse q -> 2048, dense q -> its causal length.
INDEXER_HD constexpr int publish_length(int cur_pos, int q) {
  return cur_pos + q + 1 < kTopk ? cur_pos + q + 1 : kTopk;
}
}  // namespace mega_wire

// ---------------------------------------------------------------------------
// 7. Op ids (one device function each; standalone wrapper CTA sets)
// ---------------------------------------------------------------------------
enum OpId : int {
  kOpP1 = 0,   // x-RMSNorm + W8A16 GEMV (wqakis, CTAs 0..67) and bf16 W_is GEMV (68..71)   grid 72
  kOpG2a = 1,  // ki LayerNorm + RoPE + Hadamard -> ki_cache rows cur_pos..+3 (Group2a)         grid
               // 4 (cta 0 works)
  kOpP2 =
      2,  // q_a-RMSNorm + Wqi GEMV -> IQ                                                  grid 128
  kOpP3 = 3,  // IQ RoPE + Hadamard -> IQ_RT                                                   grid
              // 4*ceil(N/16) = 8
  kOpP4 =
      4,  // tcgen05 scoring + fp16-key histogram (sparse only)                            grid 128
  kOpP5 =
      5,  // selection stage B: S x2, tickets, counting-sort scatter (sparse only)         grid 128
  kOpG2b = 6,  // selection stage C: 3-round radix refine (Group2b, 384 threads, sparse only)   grid
               // 16 (4 work)
  kOpP6 =
      7,  // ids broadcast to the peers' recv_buf                                          grid 8
  kOpP7 =
      8,  // push collective + residual + hidden                                           grid 128
  kOpCount = 9,
};
INDEXER_HD constexpr int op_cta_count(int op) {
  return op == kOpP1    ? kP1AllCtas
         : op == kOpG2a ? kGroup2aCtas
         : op == kOpP2  ? kP2Ctas
         : op == kOpP3  ? kP3CtasDefault
         : op == kOpP4  ? kP4Ctas
         : op == kOpP5  ? kP5Ctas
         : op == kOpG2b ? kGroup2bCtas
         : op == kOpP6  ? kP6Ctas
                        : kP7Ctas;
}

// Execution modes (IndexerArgs::mode; env INDEXER_MODE)
enum Mode : int {
  kModePersistent = 0,  // one 148 x 384 persistent kernel
  kModeChain = 1,       // standalone wrappers in stream order (same op bodies)
  kModeNull =
      2,  // protocol only: words, waits, ring drains, S barriers; no math (orchestration floor)
  kModeComputeOnly = 3,  // the persistent kernel with the two communications SKIPPED: every phase,
                         // boundary word, ring item and
                         //   S generation of kModePersistent, but the P6 / P7 op bodies (and CTA
                         //   0's kFlagAck pacing) are not entered -- the single-GPU compute-only
                         //   form (the communications are not exercised). Chain launches
                         //   (indexer_launch_chain) skip the P6 / P7 wrappers under it as well.
};
// The two communications run unless the mode is kModeComputeOnly (persistent driver, chain
// launcher, caller record pre-fill).
INDEXER_HD constexpr bool mode_runs_comm(int mode) { return mode != kModeComputeOnly; }

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
enum SyncWord : int {
  kSyncC0 =
      0x00,  // P1 -> P2 / Group2a          arrive +2 x 128 Group1 CTAs       wait Group1, Group2a
  kSyncC1 = 0x10,  // P2 -> P3                    +2 x 128                          wait (2nd wait
                   // dropped)
  kSyncC2 =
      0x20,  // Group2a ki_cache rows -> P4 +0x40 x 4 Group2a CTAs           wait Group1 sparse
  kSyncC3 =
      0x30,  // P3 IQ_RT -> P4              +2 x 128                          wait Group1 sparse
  kSyncC4 = 0x40,       // stage B -> Group2b          +2 x 128    wait Group2b
  kSyncS = 0x44,        // sense-reversing barrier, two generations per sparse launch
  kSyncC5 = 0x50,       // Group2b -> ids broadcast    +0x10 x 16 Group2b CTAs  wait CTAs 0..7
  kSyncC6 = 0x60,       // ids pushed -> collective    +0x20 x 8 CTAs                   wait Group1
  kSyncS1Epoch = 0x70,  // epoch-variant of S generation 1 (INDEXER_S_BARRIER_EPOCH=1 only)
  kSyncS2Epoch = 0x74,  // epoch-variant of S generation 2
  kSyncBlockBytes = 0x80,
};
constexpr uint32_t kWordSum = 0x100;  // per-launch total of every word
constexpr uint32_t kArriveC0 = 2, kArriveC1 = 2, kArriveC2 = 0x40, kArriveC3 = 2, kArriveC4 = 2,
                   kArriveC5 = 0x10, kArriveC6 = 0x20;
constexpr int kMembersC0 = kGroup1Ctas, kMembersC1 = kGroup1Ctas, kMembersC2 = kGroup2aCtas,
              kMembersC3 = kGroup1Ctas, kMembersC4 = kGroup1Ctas, kMembersC5 = kGroup2bCtas,
              kMembersC6 = kP6Ctas;
static_assert(kArriveC0 * kMembersC0 == kWordSum && kArriveC1 * kMembersC1 == kWordSum &&
                  kArriveC2 * kMembersC2 == kWordSum && kArriveC3 * kMembersC3 == kWordSum &&
                  kArriveC4 * kMembersC4 == kWordSum && kArriveC5 * kMembersC5 == kWordSum &&
                  kArriveC6 * kMembersC6 == kWordSum,
              "every boundary word sums to 0x100 per launch");
INDEXER_HD constexpr uint32_t word_target(uint32_t epoch) { return kWordSum * epoch; }
constexpr uint32_t kSenseCta0Base = 0x80000001u;  // CTA0 adds 0x80000001 - N_active, others +1

// ---------------------------------------------------------------------------
// 9. Op-descriptor ("ctx") table: 8 rows x 16 u64 (host fills 128 B per row); field offsets are
// asserted below.
// ---------------------------------------------------------------------------
struct CtxRow0 {                  // P1: RMSNormProjxWqakis + ProjxWis
  const __nv_bfloat16* x;         // +0x00 x[4][6144] bf16 (row stride 0x3000)
  const float* gamma_x;           // +0x08 RMSNorm gamma f32[6144]
  const uint8_t* wqakis_pages;    // +0x10 68 x 6 x 0x8080 (CTA stride 0x30300)
  __nv_bfloat16* q_a_out;         // +0x18 q_a[4][2048] bf16 (token stride 0x800 elems)
  __nv_bfloat16* ki_out;          // +0x20 ki[4][128] bf16 (token stride 0x80 elems)
  const uint8_t* wis_pages;       // +0x28 W_is bf16 MMA pages 2 x 6 x 0x8000
  __nv_bfloat16* idx_scores_out;  // +0x30 idx_scores[4][32] bf16 (token stride 0x40 B)
  uint64_t pad[9];
};
struct CtxRow1 {                // P2: RmsnormProjqWqi
  const __nv_bfloat16* q_a_in;  // +0x00
  const uint8_t* wqi_pages;     // +0x08 128 x 2 x 0x8080 (CTA stride 0x10100)
  uint64_t unused_0x10;         // +0x10
  const float* gamma_q;         // +0x18 RMSNorm gamma f32[2048]
  __nv_bfloat16* iq_out;        // +0x20 IQ[4][32][128] bf16 (token stride 0x2000 B)
  uint64_t pad[11];
};
struct CtxRow2 {               // Group2a: LayerNormRoPERotate (K path)
  const __nv_bfloat16* ki_in;  // +0x00 ki[4][128]
  const int32_t* cur_pos_ptr;  // +0x08
  __nv_bfloat16* ki_cache;     // +0x10 ki_cache[rows][128] (row stride 256 B)
  const float* ln_gamma;       // +0x18 LayerNorm weight f32[128]
  const float* ln_beta;        // +0x20 LayerNorm bias f32[128]
  const float* rope_cos_sin;   // +0x28 (cos,sin) interleaved f32[4][64]
  uint64_t pad[10];
};
struct CtxRow3 {                      // P3: Rotate (Q path)
  const __nv_bfloat16* iq_in;         // +0x00 IQ[4][N][128]
  __nv_bfloat16* iq_rt_out;           // +0x08 IQ_RT[4][N][128]
  const float* rope_cos_sin;          // +0x10 (cos,sin) f32, q stride 256 B
  uint64_t n_heads;                   // +0x18 N (32)
  __nv_bfloat16* second_out_base;     // +0x20 optional position-ring output (UNKNOWN use)
  const int32_t* second_out_pos_ptr;  // +0x28
  uint32_t second_out_lo_unknown;     // +0x30
  uint32_t second_out_mod;  // +0x34 modulus M of the position ring (applied only if M >= 1)
  uint64_t pad[9];
};
struct CtxRow4 {         // P4/P5: scoring + selection stage B
  uint64_t unused_0x00;  // +0x00
  const __nv_bfloat16* k_patch_src;
  const __nv_bfloat16* idx_weights;  // +0x10 idx_scores bf16[4][32]
  float* logits;
  const int32_t* cur_pos_ptr;
  uint32_t* hist_blk;  // +0x28 H block [4][258] u32, cleared by CTA0
  int32_t* selects;    // +0x30 IDX_SELECTS[4][2048]
  int32_t* cand_ws;    // +0x38 candidate ws (row stride kWsStride)
  const void* k_tmap;  // +0x40 CUtensorMap* of ki_cache [rows][128] bf16
  const void* q_tmap;  // +0x48 CUtensorMap* of IQ_RT [128][128] bf16
  uint64_t pad[6];
};
struct CtxRow5 {        // Group2b: selection stage C (aliases of row 4)
  const float* logits;  // +0x00
  int32_t* selects;     // +0x08
  const int32_t*
      cur_pos_ptr;       // +0x10 second cur_pos pointer: decides len and the 0x7fd test; the exit
                         //   gate uses row 4 +0x20.  Same value as row 4 +0x20 and
                         //   row 2 +0x08 is a host contract -- the driver traps if it does not hold
  uint64_t unused_0x18;  // +0x18
  uint32_t* hist_blk;    // +0x20
  int32_t* cand_ws;      // +0x28
  uint64_t pad[10];
};
struct CtxRow6 {           // P6: ids broadcast
  const int32_t* ids_src;  // +0x00 int32[] (= IDX_SELECTS flat [4][2048])
  const uint64_t* peer_table;
  uint32_t tag;      // +0x10 record tag (== receivers' ctx[0x928])
  uint32_t n_ids;    // +0x14 number of ids (records = n_ids / 2)
  uint32_t n_peers;  // +0x18
  uint32_t unused_0x1c;
  uint64_t pad[12];
};
struct CtxRow7 {  // P7: push collective + residual + hidden
  const __nv_bfloat16* own_slice;
  const __nv_bfloat16* residual;  // +0x08 residual x [4][6144]
  __nv_bfloat16* hidden_out;      // +0x10 hidden[4][6144] (row stride 0x3000)
  uint64_t epoch;
  uint64_t reserved_fill_f32;
  uint64_t reserved_fill_bf16;
  uint64_t pad[10];
};
struct CtxTable {
  CtxRow0 p1;
  CtxRow1 p2;
  CtxRow2 g2a;
  CtxRow3 p3;
  CtxRow4 score;
  CtxRow5 refine;
  CtxRow6 bcast;
  CtxRow7 collective;
};
constexpr int kCtxRows = 8;
constexpr int kCtxRowBytes = 128;  // host row stride
static_assert(sizeof(CtxRow0) == kCtxRowBytes && sizeof(CtxRow1) == kCtxRowBytes &&
                  sizeof(CtxRow2) == kCtxRowBytes && sizeof(CtxRow3) == kCtxRowBytes &&
                  sizeof(CtxRow4) == kCtxRowBytes && sizeof(CtxRow5) == kCtxRowBytes &&
                  sizeof(CtxRow6) == kCtxRowBytes && sizeof(CtxRow7) == kCtxRowBytes &&
                  sizeof(CtxTable) == kCtxRows * kCtxRowBytes,
              "ctx rows are 128 B");
static_assert(offsetof(CtxRow0, gamma_x) == 0x08 && offsetof(CtxRow0, wqakis_pages) == 0x10 &&
                  offsetof(CtxRow0, q_a_out) == 0x18 && offsetof(CtxRow0, ki_out) == 0x20 &&
                  offsetof(CtxRow0, wis_pages) == 0x28 && offsetof(CtxRow0, idx_scores_out) == 0x30,
              "row 0");
static_assert(offsetof(CtxRow1, wqi_pages) == 0x08 && offsetof(CtxRow1, gamma_q) == 0x18 &&
                  offsetof(CtxRow1, iq_out) == 0x20,
              "row 1");
static_assert(offsetof(CtxRow2, cur_pos_ptr) == 0x08 && offsetof(CtxRow2, ki_cache) == 0x10 &&
                  offsetof(CtxRow2, ln_gamma) == 0x18 && offsetof(CtxRow2, ln_beta) == 0x20 &&
                  offsetof(CtxRow2, rope_cos_sin) == 0x28,
              "row 2");
static_assert(offsetof(CtxRow3, iq_rt_out) == 0x08 && offsetof(CtxRow3, rope_cos_sin) == 0x10 &&
                  offsetof(CtxRow3, n_heads) == 0x18 &&
                  offsetof(CtxRow3, second_out_base) == 0x20 &&
                  offsetof(CtxRow3, second_out_pos_ptr) == 0x28 &&
                  offsetof(CtxRow3, second_out_mod) == 0x34,
              "row 3");
static_assert(offsetof(CtxRow4, k_patch_src) == 0x08 && offsetof(CtxRow4, idx_weights) == 0x10 &&
                  offsetof(CtxRow4, logits) == 0x18 && offsetof(CtxRow4, cur_pos_ptr) == 0x20 &&
                  offsetof(CtxRow4, hist_blk) == 0x28 && offsetof(CtxRow4, selects) == 0x30 &&
                  offsetof(CtxRow4, cand_ws) == 0x38 && offsetof(CtxRow4, k_tmap) == 0x40 &&
                  offsetof(CtxRow4, q_tmap) == 0x48,
              "row 4");
static_assert(offsetof(CtxRow5, selects) == 0x08 && offsetof(CtxRow5, cur_pos_ptr) == 0x10 &&
                  offsetof(CtxRow5, hist_blk) == 0x20 && offsetof(CtxRow5, cand_ws) == 0x28,
              "row 5");
static_assert(offsetof(CtxRow6, peer_table) == 0x08 && offsetof(CtxRow6, tag) == 0x10 &&
                  offsetof(CtxRow6, n_ids) == 0x14 && offsetof(CtxRow6, n_peers) == 0x18,
              "row 6");
static_assert(offsetof(CtxRow7, residual) == 0x08 && offsetof(CtxRow7, hidden_out) == 0x10 &&
                  offsetof(CtxRow7, epoch) == 0x18 &&
                  offsetof(CtxRow7, reserved_fill_f32) == 0x20 &&
                  offsetof(CtxRow7, reserved_fill_bf16) == 0x28,
              "row 7");
static_assert(offsetof(CtxTable, p2) == 0x80 && offsetof(CtxTable, g2a) == 0x100 &&
                  offsetof(CtxTable, p3) == 0x180 && offsetof(CtxTable, score) == 0x200 &&
                  offsetof(CtxTable, refine) == 0x280 && offsetof(CtxTable, bcast) == 0x300 &&
                  offsetof(CtxTable, collective) == 0x380,
              "ctx table rows");

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
struct alignas(64) TensorMap128 {  // CUtensorMap by value (128 B, 64 B aligned); opaque here so the
  uint64_t opaque[16];  // host header does not need cuda.h (encode with cuTensorMapEncodeTiled)
};
static_assert(sizeof(TensorMap128) == 128, "CUtensorMap is 128 B");

struct alignas(64) IndexerArgs {
  const CtxTable* ctx_table;
  uint32_t* sync_words;
  uint8_t* peers[kWorld];
  uint32_t n_rows;
  int32_t rank;
  // ---- partition 2: MegaArgs family -- names and semantics of the attention kernel's
  // mega_attn_abi.h; the host fills them
  //      the same way ----
  uint32_t epoch;
  int32_t cur_pos;     // +0x184  host snapshot of cur_pos, -1 if unknown (chain then launches the
                       // sparse ops too).  The DEVICE value is
                       //        *cur_pos_dev if set, else *ctx row 4 +0x20 -- never this field
  int32_t mode;        // +0x188  Mode (persistent | chain | null | compute_only)
  int32_t ids_direct;  // +0x18c  direct / loopback: the peers neither wait for kFlagPub nor write
                       // kFlagAck (consume_selected_ids returns at
                       //        once) -> caller forces skip_done_wait = 1 with it; with dense_ids
                       //        the publish is skipped
  int32_t skip_done_wait;  // +0x190  1: CTA 0 skips the seven kFlagAck waits; implied by ids_direct
                           // (nobody would ever ack)
  int32_t single_rank;     // +0x194  1: loopback -- peers[0..7] alias the local arena, CTA 0 alone
                           // publishes to peers[rank], no ack wait
  int32_t rank0_reduce;    // +0x198  1: P7 reduces the seven kHiddenRx slices into the endpoint; 0:
                           // P7 only fills the NaN sentinels.
                           //        Must be 1 on BOTH sides: the attention kernel pushes records to
                           //        peers[0] (-4856); a form that never does needs 0 there (caller
                           //        INDEXER_PEER_FORM), else P7 polls forever
  int32_t dense_ids;  // +0x19c  replace mode, ctx <= 2048: the pure ranks self-generate ids[q][i] =
                      // i, lengths = min(2048, cur_pos+q+1);
                      //        with ids_direct rank 0 skips the publish .  Alone it changes nothing
                      //        on rank 0 (we always publish lengths)
  const int32_t* cur_pos_dev;  // +0x1a0 if set: cur_pos = *cur_pos_dev at kernel entry
  const uint32_t* epoch_dev;   // +0x1a8  if set (and epoch_ctr is not): epoch =
                               // atomicAdd(*epoch_dev, 1) / kGrid + 1 at entry (launch ordinal)
  uint32_t* epoch_ctr;         // +0x1b0  if set: per-CTA launch counters u32[kGrid], epoch =
                               // ++epoch_ctr[cta] (plain load/store); precedence over
  //        epoch_dev; persistent mode only (a chain launch would count 9 per
  //        layer -- caller refuses)
  uint64_t reserved_trace;
  //        [1 + (epoch & 4095)] = cur_pos; every CTA's thread 0 the raw-entry / entry (atomicMin)
  //        and exit (atomicMax) stamps
  uint64_t reserved_fill_f32;
  uint64_t reserved_fill_bf16;
  __nv_bfloat16*
      endpoint;  // +0x1d0  override of ctx row 7 +0x10 hidden_out (bf16[4][6144]); nullptr -> ctx
  const __nv_bfloat16* x;  // +0x1d8  override of ctx row 7 +0x08 residual (== row 0 +0x00 x);
                           // nullptr -> the ctx pointer
  // ---- partition 3: indexer implementation private ----
  uint8_t* arena;  // +0x1e0 local arena base; arena::* offsets are absolute in it (compat: ==
                   // peers[rank], caller asserts)
  const int32_t* cur_pos_ptr;  // +0x1e8 == ctx_table->score.cur_pos_ptr (host tools; the kernel
                               // reads the ctx row / cur_pos_dev)
  uint32_t flags;              // +0x1f0 ArgFlags
  uint32_t sparse_epoch;  // +0x1f4 number of sparse launches incl. this one (bookkeeping / trace)
  uint32_t s_cum_participants;  // +0x1f8 SUM over all sparse launches so far (incl. this one) of
                                // n_active_ctas(cur_pos): wait target of the
                                //        INDEXER_S_BARRIER_EPOCH variant (cumulative, not n_active
                                //        x epoch; host value, graph replay incompatible)
  uint32_t inline_ctx;  // +0x1fc: compute-only run carries its context in the launch parameters
  uint64_t inline_rows[6][8];  // +0x200..+0x37f: live 64-B prefixes of compute rows 0..5
  TensorMap128 k_tmap;         // diagnostic inline-context mode only
  TensorMap128 q_tmap;
};
INDEXER_HD constexpr bool has_inline_ctx(const IndexerArgs& args) {
  return args.inline_ctx != 0 && args.mode == kModeComputeOnly && args.cur_pos >= 0;
}
INDEXER_HD constexpr bool has_inline_dense_ctx(const IndexerArgs& args) {
  return has_inline_ctx(args) && !is_sparse(args.cur_pos);
}
enum ArgFlags : uint32_t {
  kFlagDenseSkipBroadcast = 1u << 0,  // dense launch: CTAs 0..7 push no records
  kFlagPrefetchWaitC2 = 1u << 1,  // prefetchers wait C2 before the K tiles that cover cur_pos..+3.
                                  //   PERSISTENT mode only: in a chain no wrapper arrives on C2,
                                  //   the P4 wrapper's prefetch thread would poll the word forever
                                  //   -- caller refuses the flag with INDEXER_MODE=chain
  kFlagNativePitch = 1u << 2,  // logits / cand_ws rows use the pitch L = cur_pos + 4 instead of the
                               // fixed kLogitsStride /
                               //   kWsStride; default off = fixed pitch.
};
// Row strides of the two L-strided scratch buffers.  P4 writes logits, P5 + G2b read them; P5
// writes cand_ws, G2b reads it.  Every op that touches either buffer MUST take its stride from
// these two helpers, so one flag flips the whole kernel between the fixed pitch and the cur_pos + 4
// pitch.  Both arena regions hold 4 x L for every L <= kMaxSeqLen.
INDEXER_HD constexpr int logits_row_stride(uint32_t flags, int cur_pos) {
  return (flags & kFlagNativePitch) ? total_len(cur_pos) : kLogitsStride;
}
INDEXER_HD constexpr int ws_row_stride(uint32_t flags, int cur_pos) {
  return (flags & kFlagNativePitch) ? total_len(cur_pos) : kWsStride;
}
static_assert(kLogitsStride >= kMaxSeqLen && kWsStride >= kMaxSeqLen,
              "native pitch L = cur_pos + 4 <= kMaxSeqLen must fit the fixed-stride arena rows");
static_assert(offsetof(IndexerArgs, ctx_table) == 0 && offsetof(IndexerArgs, peers) == 16,
              "named indexer launch parameters");
// host convention (caller): INDEXER_PEERS_INBOX_BASES == 0: args.peers[r] = peer arena r (the
// kernel adds the mega_wire offsets and

// range never overlaps the sync / H / peer-table / tensor-map / selects regions that precede it
// (checked in namespace arena).

// ---------------------------------------------------------------------------

//   int32 units unless noted:
//   [0] execution count (= the epoch of the last launch), [1 + (epoch & 4095)] cur_pos history CTA
//   0 thread 0 u64 stamps at int32 index 4098: [0..4095] entry = min over CTAs right after the
//   epoch / cur_pos load, [4096..8191] exit = max

//   launch count & 4095.  The host allocates 4098*4 + 8192*8 + 4096*8 + 4096*8 + 256*4 = 148,488 B
//   and memsets the entry / raw stamps to 0xFF; a reader takes the first 4098*4 + 8192*8 + 4096*8 B
//   (duration = exit - entry, prologue = entry - raw per epoch).

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 11. Global arena of rank 0 (one cudaMalloc, IPC-exported; absolute byte offsets)
// [0, mega_wire::kIdsRx + kIdsRxBytes) is the mega layout byte for byte -- the pure ranks address
// rank 0's arena by those offsets (peers[0] + k5_rx_offset(...) -> kHiddenRx records) and rank 0
// addresses THEIR arenas by the same constants (kIdsRx / kFlagPub / kFlagAck); selects is our
// IDX_SELECTS copy (P6 ids source); the layout is kept in both builds.  The pure ranks allocate
// mega_wire::kArenaBytes (8 MiB), rank 0 arena::kArenaBytes (16 MiB): an IPC handle covers the
// exporter's whole allocation.
// ---------------------------------------------------------------------------
namespace arena {
constexpr size_t kIndexerBase =
    (mega_wire::kIdsRx + mega_wire::kIdsRxBytes + 127) & ~size_t(127);  // 0x728700 = 7,505,664
constexpr size_t kSync = kIndexerBase + 0x0000;
constexpr size_t kHist = kIndexerBase + 0x0080;  // u32[4][258] H block (q stride 0x408)
constexpr size_t kHistBytes = size_t(kSeqLen) * kHistBlockBytes;  // 0x1020
constexpr size_t kPeerIdsTable = kIndexerBase + 0x1100;           // u64[8] recv_buf pointers
constexpr size_t kTmaps =
    kIndexerBase + 0x1180;  // 2 x 128 B CUtensorMap (k, q), 64 B aligned (row 4 +0x40/+0x48)
constexpr size_t kEpochCtr =
    kIndexerBase + 0x1400;  // u32[kGrid] per-CTA epoch counters (IndexerArgs::epoch_ctr may point
                            // here; zeroed with the arena)
constexpr size_t kEpochCtrBytes = size_t(kGrid) * 4;  // 592
constexpr size_t kSelects =
    kIndexerBase + 0x2000;  // int32[4][2048] IDX_SELECTS (P5/G2b write, P6 reads)
constexpr size_t kSelectsBytes = size_t(kSeqLen) * kTopk * 4;  // 0x8000
constexpr size_t kInbox = kIndexerBase + 0x10000;              // args.peers[r] = arena_r + kInbox
constexpr size_t kInboxBytes = ::flashinfer::glm5_tp8::indexer_q4::kInboxBytes;  // 0xE0000
constexpr size_t kLogits = kIndexerBase + 0xF0000;                         // f32[4][kLogitsStride]
constexpr size_t kLogitsBytes = size_t(kSeqLen) * kLogitsStride * 4;       // 3,244,160
constexpr size_t kWs = kLogits + ((kLogitsBytes + 4095) & ~size_t(4095));  // int32[4][kWsStride]
constexpr size_t kWsBytes = size_t(kSeqLen) * kWsStride * 4;               // 3,280,960
constexpr size_t kEnd = kWs + kWsBytes;
constexpr size_t kTrace = (kEnd + 127) & ~size_t(127);
constexpr size_t kP3HeadCounters = kTrace;
constexpr int kTraceStamps = 64;
constexpr int kTraceEpochSlots = 16;
constexpr size_t kTraceCtaBytes = size_t(kTraceStamps) * 8;
constexpr size_t kTraceEpochBytes = size_t(kGrid) * kTraceCtaBytes;
constexpr size_t kTraceBytes = size_t(kTraceEpochSlots) * kTraceEpochBytes;
constexpr size_t kArenaBytes =
    size_t(16) * 1024 *
    1024;  // 16 MiB (rank 0); the pure ranks allocate mega_wire::kArenaBytes = 8 MiB
constexpr int kCompScratchCap = 20480;
constexpr size_t kCompScratch = kTrace + 4096;
constexpr size_t kCompScratchBytes = size_t(kSeqLen) * kCompScratchCap * 8;
static_assert(kP3HeadCounters + kIdxHeads * 4 <= kCompScratch && kCompScratch % 128 == 0 &&
                  kCompScratch + kCompScratchBytes <= kTrace + kTraceBytes,
              "refine composites fit the unused trace region");
static_assert(kIndexerBase % 128 == 0 &&
                  kIndexerBase >= mega_wire::kIdsRx + mega_wire::kIdsRxBytes &&
                  kIndexerBase == 0x728700,
              "indexer-private block starts right after the attention ids inbox (kIdsRx + 2 x "
              "kIdsRxParityBytes), 128 B aligned");
static_assert(kSync + kSyncBlockBytes <= kHist && kHist + kHistBytes <= kPeerIdsTable &&
                  kPeerIdsTable + 8 * kWorld <= kTmaps && kTmaps + 256 <= kEpochCtr &&
                  kEpochCtr + kEpochCtrBytes <= kSelects && kSelects + kSelectsBytes <= kInbox &&
                  kInbox + kInboxBytes <= kLogits && kLogits + kLogitsBytes <= kWs &&
                  kEnd <= kTrace && kTrace + kTraceBytes <= kArenaBytes,
              "arena regions overlap / overflow");
static_assert(kSync % 128 == 0 && kHist % 128 == 0 && kTmaps % 64 == 0 && kEpochCtr % 128 == 0 &&
                  kSelects % 128 == 0 && kInbox % 128 == 0 && kLogits % 128 == 0 && kWs % 128 == 0,
              "arena alignment");
static_assert(kArenaBytes >= mega_wire::kArenaBytes &&
                  mega_wire::kHiddenRx + mega_wire::kHiddenRxBytes <= kIndexerBase,
              "rank 0's arena contains the whole pure-rank layout; the K5 records the pure ranks "
              "push into it (kHiddenRx) end below the "
              "indexer-private block");
}  // namespace arena
// what a device push to a peer's INBOX adds to args.peers[dst]: nothing when caller already hands
// over inbox bases, arena::kInbox when peers[] are arena bases (any compat wire, MegaArgs::peers
// semantics).
constexpr size_t kPeersInboxOffset = INDEXER_PEERS_INBOX_BASES ? size_t(0) : arena::kInbox;

// ---------------------------------------------------------------------------
// 12. Shared-memory map.  Offsets are USER offsets into our single dynamic buffer (`extern
// __shared__ __align__(1024) The window's first 1 KB [0, 0x400) is the driver/ptxas reserve (the
// TMEM alloc bookkeeping words at window 0x54/0x58/0x5c live there and are managed by the
// tcgen05.alloc code ptxas emits) -- it is NOT part of the buffer we allocate.

// We request the same 0x38000 B as one dynamic buffer (same footprint / carveout).
// ---------------------------------------------------------------------------
namespace smem {
constexpr uint32_t kCtx = 0x0000;  // ctx table rows, stride 0x1c0 (window 0x400 + r*0x1c0)
constexpr uint32_t kCtxRowStride = 0x01c0;
constexpr uint32_t kCtxEnd = kCtx + kCtxRows * kCtxRowStride;  // 0x0e00
constexpr uint32_t kRmsPartial = 0x3800;  // RMSNorm 8-warp partials (0x80 B) (window 0x3c00)
constexpr uint32_t kFullBar = 0x3900;
constexpr uint32_t kEmptyBar = 0x3980;    // empty[4] mbarriers (+0x3980)
constexpr uint32_t kEpochWord = 0x3a00;   // u32 device-loaded epoch: thread 0 writes it in the
                                          // prologue, every role reads it after the barrier
constexpr uint32_t kCurPosWord = 0x3a04;  // i32 cur_pos = *cur_pos_dev (compat) or *ctx row 4 +0x20
constexpr uint32_t kRing =
    0x3c00;  // 4 x 0x9000 page ring (window 0x4000, 1 KB aligned in the window)
constexpr uint32_t kPubStage = kRing;  // compat P6 (INDEXER_PUBLISH_BULK=1): the 32,784 B publish
                                       // payload is staged in ring slot 0 (idle after P4)
constexpr uint32_t kRingEnd = kRing + kRingBytes;  // 0x27c00
constexpr uint32_t kAct =
    0x27c00;  // phase-dependent activation / tile region (union, 0x8000 B; window 0x28000)
// P1 (CTAs 0..67): x_norm bf16 [4][6144] token stride 0x3010 (16 B skew), output, K-split reduction
constexpr uint32_t kP1XnormTokStride = 0x3010;
constexpr uint32_t kP1Out = 0x33c80;  // bf16 [4][32] (window 0x34080)
constexpr uint32_t kP1Red =
    0x33d80;  // f32 [8 warps][0x404 B]: warp*0x404 + (lane&3)*0x100 + (lane>>2)*4 (window 0x34180)
constexpr uint32_t kP1RedWarpStride = 0x0404;
// P1b (CTAs 68..71): x_norm token stride 0x3000, output window 0x2e000, reduction window 0x2e080
constexpr uint32_t kP1bXnormTokStride = 0x3000;
constexpr uint32_t kP1bOut = 0x2dc00;
constexpr uint32_t kP1bRed = 0x2dc80;
constexpr uint32_t kP1bRedWarpStride = 0x0084;
// P2 (all 128): qa_norm token stride 0x1020, output window 0x2c080, reduction window 0x2c180
constexpr uint32_t kP2QnormTokStride = 0x1020;
constexpr uint32_t kP2Out = 0x2bc80;
constexpr uint32_t kP2Red = 0x2bd80;
constexpr uint32_t kP2RedWarpStride = 0x0204;
// P4 scoring: q tile 8 boxes (dim half h -> +h*0x4000, row block r -> +r*0x1000), TMEM base, 2
// count-1 mbarriers, histogram
constexpr uint32_t kQTile = 0x27c00;
constexpr uint32_t kQTileHalfStride = 0x4000;
constexpr uint32_t kQTileRowBlockStride = 0x1000;
constexpr uint32_t kTmemBase = 0x2fc00;  // u32 column base (window 0x30000)
constexpr uint32_t kMmaBar = 0x2fc10;    // tcgen05 commit mbarrier, count 1 (window 0x30010)
constexpr uint32_t kMmaBar1 = 0x2fc18;
constexpr uint32_t kQTmaBar = 0x2fc20;  // q tile TMA mbarrier, count 1 (window 0x30020)
constexpr uint32_t kHist =
    0x2fc30;  // u32 [4][258] local histogram (q stride 0x408) (window 0x30030)
constexpr uint32_t kHistEnd = kHist + kSeqLen * kHistBlockBytes;  // 0x30c50
constexpr uint32_t kP5BinCache =
    kHistEnd;  // packed u8 bin pair [3 K tiles][2 query pairs][128 rows]
constexpr uint32_t kP5BinCacheTiles = 3;
constexpr uint32_t kP5BinCacheEnd =
    kP5BinCache + kP5BinCacheTiles * (kSeqLen / 2) * kKTileRows * 4;  // 0x31850
constexpr uint32_t kThrBin = 0x31c70;    // u16 [4] threshold bin (window 0x32070)
constexpr uint32_t kSureBase = 0x31c80;  // u32 [4] (window 0x32080)
constexpr uint32_t kCandBase = 0x31ca0;  // u32 [4] (window 0x320a0)
constexpr uint32_t kP2HeadFlag = 0x31cc0;

constexpr uint32_t kBin8 = 0x31d00;
constexpr uint32_t kBin8Tiles = 13;
constexpr uint32_t kBin8End = kBin8 + kBin8Tiles * kSeqLen * kKTileRows;
// Group2a (CTA 128): slot 0 holds gamma/beta/cos-sin, activation region holds ki and the result
constexpr uint32_t kG2aGamma = 0x3c00;   // f32[128] (window 0x4000)
constexpr uint32_t kG2aBeta = 0x3e00;    // f32[128]
constexpr uint32_t kG2aCosSin = 0x4000;  // f32[4][64]
constexpr uint32_t kG2aKi = 0x27c00;
constexpr uint32_t kG2aResult = 0x28c00;  // bf16 [4][128] (window 0x29000)
// Group2b (CTAs 132..135): threshold, second-level histogram, two counters, two candidate lists
constexpr uint32_t kG2bThr = 0x3c00;    // u16 (window 0x4000)
constexpr uint32_t kG2bHist2 = 0x3c10;  // u32 [257] (window 0x4010)
constexpr uint32_t kG2bCnt = 0x4020;    // u32 [2] (+0 / +4) (window 0x4420)
constexpr uint32_t kG2bList0 = 0x4030;  // int32 [16384] (window 0x4430)
constexpr uint32_t kG2bList1 = 0x14030;
constexpr uint32_t kG2bEnd = kG2bList1 + kRefineMaxCand * 4;  // 0x24030
// P7 collective: own slice, residual, received records
constexpr uint32_t kP7Slice = 0x27c00;     // bf16 [4][48] (window 0x28000)
constexpr uint32_t kP7Residual = 0x27f00;  // (window 0x28300)
constexpr uint32_t kP7Rx = 0x28200;        // + q*0x2a0 + slot*0x60 (window 0x28600)
constexpr uint32_t kP7RxQStride = 0x02a0;
constexpr uint32_t kP7RxSlotStride = 0x0060;
constexpr uint32_t kLogitCache = kBin8 + kBin8Tiles * kSeqLen * kKTileRows;
constexpr uint32_t kLogitCacheTiles = 9;
constexpr uint32_t kLogitCacheEnd = kLogitCache + kLogitCacheTiles * kSeqLen * kKTileRows * 4;
constexpr uint32_t kUsedEnd = 0x35da0;
constexpr uint32_t kDynamicBytes = 0x38000;  // our single dynamic buffer
static_assert(kLogitCache % 16 == 0 && kLogitCacheEnd <= kDynamicBytes,
              "P4 logit cache fits behind the bin cache");
static_assert(kP1Red + kConsumerWarps * kP1RedWarpStride <= kUsedEnd && kUsedEnd <= kDynamicBytes,
              "P1 reduction fits; used extent inside the buffer");
static_assert(kCtxEnd <= kRmsPartial && kRmsPartial + 0x80 <= kFullBar &&
                  kEmptyBar + 8 * kRingSlots <= kRing,
              "static region layout");
static_assert(kEmptyBar + 8 * kRingSlots <= kEpochWord && kCurPosWord + 4 <= kRing,
              "epoch / cur_pos words live in the [0x39a0, 0x3c00) gap");
static_assert(kPubStage % 16 == 0 &&
                  kPubStage + mega_wire::kPublishPayloadBytes <= kRing + kRingSlotBytes,
              "publish payload fits ring slot 0");
static_assert(kG2bEnd <= kRingEnd && kP5BinCacheEnd <= kThrBin && kQTile + 8 * 0x1000 <= kTmemBase,
              "smem unions");
static_assert(kP2HeadFlag >= kCandBase + 4 * kSeqLen && kP2HeadFlag + 4 <= kBin8 &&
                  kMmaBar1 + 8 <= kQTmaBar,
              "fused-P3 flag / second MMA barrier");
static_assert(kBin8 >= kCandBase + 4 * kSeqLen && kBin8End <= kP1Out &&
                  kBin8Tiles * kKTileStrideRows >= kMaxSeqLen,
              "stage-A bin cache: between the P5 scalars and the (earlier) P1 output, covers every "
              "tile of L <= kMaxSeqLen");
static_assert(2 * kQTileHalfStride == kQTileBytes && 4 * kQTileRowBlockStride == kQTileHalfStride &&
                  kQTile + kQTileBytes <= kTmemBase,
              "q tile: 2 halves x 4 row blocks x 4 KB = 0x8000 (expect_tx )");
static_assert(umma_smem_desc_lo(0x400 + kQTile) == 0x12800u &&
                  umma_smem_desc_lo(0x400 + kRing) == 0x10400u &&
                  umma_smem_desc_lo(0x400 + kRing + 3 * kRingSlotBytes) == 0x11f00u,
              "UMMA smem descriptor lo of the q tile and of the ring slots");
static_assert((0x400 + kRing) % 1024 == 0 && kRingSlotBytes % 1024 == 0 &&
                  (0x400 + kQTile) % 1024 == 0,
              "TMA / UMMA SWIZZLE_128B: 1 KB aligned in the shared WINDOW (user offset + 0x400)");
static_assert(kDynamicBytes <= 232448, "exceeds sm_100 max dynamic smem per block (227 KB)");
static_assert(kAct + 4 * kP1XnormTokStride <= kP1Out, "P1 x_norm rows fit before the P1 output");
static_assert(kAct + 2 * kP1bXnormTokStride <= kP1bOut &&
                  kP1bRed + kConsumerWarps * kP1bRedWarpStride <= kTmemBase,
              "P1b x_norm (2 tokens) ends at the P1b output; P1b reduction fits (A1 audit)");
static_assert(kAct + kSeqLen * kP2QnormTokStride <= kP2Out &&
                  kP2Red + kConsumerWarps * kP2RedWarpStride <= kTmemBase,
              "P2 qa_norm (4 tokens) ends at the P2 output; P2 reduction fits (A1 audit)");
}  // namespace smem

#ifdef __CUDACC__
// ---------------------------------------------------------------------------
// 13. Device-side per-CTA context (data only)
// ---------------------------------------------------------------------------
struct OpCtx {
  int op;   // OpId
  int cta;  // index within the op's CTA set (persistent: blockIdx.x or blockIdx.x - group base)
  int num_ctas;      // size of the op's CTA set
  int tid;           // threadIdx.x
  int warp;          // tid >> 5
  int lane;          // tid & 31
  int sync_threads;  // 256 (consumer group, bar.sync 4,0x100) or 384 (Group2b, bar.sync 4,0x180)
  bool prefetcher;   // warp >= 8
  bool null_math;    // kModeNull: protocol only
  bool sparse;       // cur_pos >= 0x7fc
  int32_t cur_pos;   // smem::kCurPosWord: *cur_pos_dev or *ctx[4].cur_pos_ptr, loaded once in the
                     // prologue
  uint32_t epoch;
  uint8_t* smem;    // dynamic smem base (offsets smem::*)
  uint32_t smem_s;  // shared-window address of smem
  uint8_t* arena;   // args.arena
  uint32_t* sync;   // args.sync_words (byte offsets kSync*)
  uint32_t item;    // next page-ring item of this CTA (monotonic across ops; == prefetcher order)
  unsigned long long* trace;
};

__device__ __forceinline__ uint32_t smem_u32(const void* generic) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(generic));
}
// ctx row R as copied into smem.
template <typename Row>
__device__ __forceinline__ const Row& ctx_row(const uint8_t* smem_base, int row) {
  return *reinterpret_cast<const Row*>(smem_base + smem::kCtx + row * smem::kCtxRowStride);
}
__device__ __forceinline__ const CtxRow0& ctx_p1(const uint8_t* s) {
  return ctx_row<CtxRow0>(s, 0);
}
__device__ __forceinline__ const CtxRow1& ctx_p2(const uint8_t* s) {
  return ctx_row<CtxRow1>(s, 1);
}
__device__ __forceinline__ const CtxRow2& ctx_g2a(const uint8_t* s) {
  return ctx_row<CtxRow2>(s, 2);
}
__device__ __forceinline__ const CtxRow3& ctx_p3(const uint8_t* s) {
  return ctx_row<CtxRow3>(s, 3);
}
__device__ __forceinline__ const CtxRow4& ctx_score(const uint8_t* s) {
  return ctx_row<CtxRow4>(s, 4);
}
__device__ __forceinline__ const CtxRow5& ctx_refine(const uint8_t* s) {
  return ctx_row<CtxRow5>(s, 5);
}
__device__ __forceinline__ const CtxRow6& ctx_bcast(const uint8_t* s) {
  return ctx_row<CtxRow6>(s, 6);
}
__device__ __forceinline__ const CtxRow7& ctx_collective(const uint8_t* s) {
  return ctx_row<CtxRow7>(s, 7);
}
template <typename T>
__device__ __forceinline__ T* smem_at(uint8_t* smem_base, uint32_t offset) {
  return reinterpret_cast<T*>(smem_base + offset);
}
__device__ __forceinline__ uint32_t* sync_word(const OpCtx& ctx, int word_offset) {
  return reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(ctx.sync) + word_offset);
}
#endif  // __CUDACC__

}  // namespace flashinfer::glm5_tp8::indexer_q4
