/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "q1_launch.cuh"

namespace flashinfer::glm5_tp8::mla_q1 {

constexpr int kWorld = 8;
constexpr int kGrid = 148;  // launch grid; CTA roles are by blockIdx.x, so the grid MUST be 148
constexpr int kNativeBlock =
    384;  // block; the kernel accepts any blockDim.x in [256, 1024] (multiple of 32)
constexpr int kNativeDynSmem =
    214016;  // dynamic smem; the kernel needs far less but tolerates this value (opt-in in init)
constexpr int kNRows = 8;
constexpr int kRowBytes = 128;  // op-table row stride in global memory (16 x u64)

// ---------------------------------------------------------------------------------------------------------------
// Compact operator parameters; mutable state is owned by the caller.
using LaunchArgs = attention_q1::Q1Args;

//    residue and are never read.
// ---------------------------------------------------------------------------------------------------------------
struct PureRow0 {  // P1: x-RMSNorm + QKVA GEMV (CTAs 0..81), k_pe raw -> pe_cache[cur_pos + t]
  const uint16_t* x;
  const float* x_gamma;  // +0x08 f32 [6144]
  const uint8_t* qkva;   // +0x10 fp8 pages [82 CTA][6][0x8080] (16,184,832 B)
  const int32_t* cur_pos;
  uint16_t* post_q;    // +0x20 bf16 [S][2048]   Q(0)
  uint16_t* post_kv;   // +0x28 bf16 [S][512]    KV(1)
  uint16_t* pe_cache;  // +0x30 bf16 [rows][64]  row cur_pos + t receives the raw k_pe (RoPE'd in
                       // place by the tail CTA)
  uint64_t unused[9];  // +0x38..+0x7f
};
struct PureRow1 {                // P2: q-RMSNorm + WqB (CTAs 0..79)
  const uint16_t* alias_post_q;  // +0x00 = row0.post_q (Q in)
  const uint8_t* wqb;            // +0x08 fp8 pages [80 CTA][2][0x8080] (5,263,360 B)
  uint64_t reserved_0x10;
  const float* q_gamma;  // +0x18 f32 [2048]
  uint16_t* post_qnd;
  uint16_t* post_qpe;
  uint64_t unused[10];  // +0x30..+0x7f
};
struct PureRow2 {  // tail RoPE (CTA 132 + t): Q_PE in place, pe_cache[cur_pos + t] in place
  uint16_t* alias_post_qpe;  // +0x00 = row1.post_qpe
  uint16_t* alias_pe_cache;  // +0x08 = row0.pe_cache
  const float* rope;  // +0x10 f32 [S][64] interleaved (cos, sin) pairs (ROPE_FREQS(30) row t ==
                      // freqs_table[cur_pos + t])
  const int32_t* alias_cur_pos;  // +0x18 = row0.cur_pos
  uint64_t heads_rope;           // +0x20 = 10
  uint64_t unused[11];           // +0x28..+0x7f
};
struct PureRow3 {  // tail kv-norm (CTA 128 + t): kv_cache[cur_pos + t] = bf16(kv * (gamma *
                   // rsqrt(fma(sumsq, 2^-9, 1e-5))))
  const uint16_t* alias_post_kv;  // +0x00 = row0.post_kv (c_kv in)
  const float* kv_gamma;          // +0x08 f32 [512]
  uint16_t* kv_cache;             // +0x10 bf16 [rows][512]
  const int32_t* alias_cur_pos;   // +0x18 = row0.cur_pos
  uint64_t unused[12];            // +0x20..+0x7f
};
struct PureRow4 {                  // P3: ProjQ (CTAs 0..79; 64 rows each)
  const uint16_t* alias_post_qnd;  // +0x00 = row1.post_qnd
  const uint8_t* projq;            // +0x08 fp8 pages [80 CTA][0x3080] (993,280 B)
  uint64_t unused_0x10;            // +0x10 (0)
  uint16_t* post_qnope;
  uint64_t unused[12];  // +0x20..+0x7f
};
struct PureRow5 {  // P4/P5: split-KV attention (q1: CTAs 0..31, split = ctaid) + combine (q1: CTAs
                   // 0..9 = head)
  const uint16_t* alias_post_qnope;  // +0x00 = row4.post_qnope
  const uint16_t* alias_post_qpe;    // +0x08 = row1.post_qpe (after the RoPE tail)
  const uint16_t* alias_kv_cache;    // +0x10 = row3.kv_cache
  const uint16_t* alias_pe_cache;    // +0x18 = row0.pe_cache
  const int32_t* ids;  // +0x20 int32 [S][2048] local ids table IDX_SELECTS(9): read only if ll_recv
                       // == 0 and kv_len > 2048
  uint16_t* post_o;    // +0x28 bf16 [S][10][512]  O(11)        (q4 token stride 0x2800 B)
  float* o_acc;
  float* lse;
  float* lse_acc;  // +0x40 f32 [S][32 split][12]  O_LSE_ACC(14): 10 live + 2 pad (pad never
                   // written); token stride 0x600 B
  const int32_t* alias_cur_pos;  // +0x48 = row0.cur_pos
  uint64_t heads_attn;           // +0x50 = 10
  uint64_t tmap_idx;             // +0x58 tensormap select (>= 2 -> 0)
  const int32_t* ll_recv;  // +0x60 NCCL-LL rows recv_buf [8*2048*2] int32 = {id0, tag, id1, tag} 16
                           // B records, record r = (q*32 + split)*32 + lane;
                           //       0 -> use the local ids table when kv_len > 2048
  uint64_t ll_tag;
  uint64_t reserved[2];  // +0x70..+0x7f
};
struct PureRow6 {                // P6: ProjO (CTAs 0..79; 32 rows each)
  const uint16_t* alias_post_o;  // +0x00 = row5.post_o
  const uint8_t* projo;          // +0x08 fp8 pages [80 CTA][0x4080] (1,320,960 B)
  uint64_t unused_0x10;          // +0x10 (0)
  uint16_t* post_projo;
  uint64_t unused[12];  // +0x20..+0x7f
};
struct PureRow7 {  // P7/P8: UnProj (128 CTAs x 48 rows) + K5 push collective + residual + NaN
                   // side-buffer refills
  const uint16_t* alias_post_projo;  // +0x00 = row6.post_projo
  const uint8_t* unproj;             // +0x08 fp8 pages [128 CTA][5][0x6080] (15,810,560 B)
  const float* out_scale;
  uint16_t* hidden_out;
  const uint16_t* residual;
  uint64_t epoch;  // +0x28 u64: low 32 bits = collective record tag E, parity = E & 1
  uint64_t reserved_fill_f32;
  uint64_t reserved_fill_bf16;
  uint64_t reserved[8];  // +0x40..+0x7f
};
static_assert(sizeof(PureRow0) == kRowBytes && sizeof(PureRow1) == kRowBytes &&
                  sizeof(PureRow2) == kRowBytes && sizeof(PureRow3) == kRowBytes &&
                  sizeof(PureRow4) == kRowBytes && sizeof(PureRow5) == kRowBytes &&
                  sizeof(PureRow6) == kRowBytes && sizeof(PureRow7) == kRowBytes,
              "PureMla rows are 128 B");
static_assert(offsetof(PureRow0, pe_cache) == 0x30 && offsetof(PureRow1, post_qpe) == 0x28 &&
                  offsetof(PureRow2, rope) == 0x10 && offsetof(PureRow3, kv_cache) == 0x10 &&
                  offsetof(PureRow4, post_qnope) == 0x18 && offsetof(PureRow5, ids) == 0x20 &&
                  offsetof(PureRow5, o_acc) == 0x30 && offsetof(PureRow5, lse_acc) == 0x40 &&
                  offsetof(PureRow5, ll_recv) == 0x60 && offsetof(PureRow5, ll_tag) == 0x68 &&
                  offsetof(PureRow6, post_projo) == 0x18 &&
                  offsetof(PureRow7, hidden_out) == 0x18 && offsetof(PureRow7, epoch) == 0x28 &&
                  offsetof(PureRow7, reserved_fill_bf16) == 0x38,
              "PureMla field offsets (protocol 2.1)");
struct PureTable {
  PureRow0 r0;
  PureRow1 r1;
  PureRow2 r2;
  PureRow3 r3;
  PureRow4 r4;
  PureRow5 r5;
  PureRow6 r6;
  PureRow7 r7;
};
static_assert(sizeof(PureTable) == kNRows * kRowBytes, "8 rows x 128 B");

// ---------------------------------------------------------------------------------------------------------------
// 3. SparseSelect (rank 0) rows -- documentation only.  The kernel does not

//      row0: +0x00 x (EH_PROJ) | +0x08 x_gamma | +0x10 idx_qkva (68 CTA x 6 pages: rows 0..2047
//      W_qa, 2048..2175 W_ki) | +0x18 post_q (Q(0))

//      row1: +0x08 idx_wqb (W_qi, 128 CTA x 2 pages) | +0x18 q_gamma | +0x20 idx_q (IQ(5))
//      row2: +0x08 cur_pos | +0x10 idx_cache [rows][128] | +0x18/+0x20 LayerNorm gamma/beta
//      f32[128] | +0x28 rope row3: +0x08 idx_q_rope (IQ_RT(6)) row4: +0x08 K-cache patch source |
//      +0x18 idx_scores (IDX_LOGITS(8), row stride L) | +0x20 cur_pos | +0x28 idx_scratch
//      (IDX_SEL_WS(42))
//            | +0x30 ids (IDX_SELECTS(9))
//      row6: +0x00 ids | +0x08 id_peers int64[7] (recv_buf of the pure ranks) | +0x10 low 32 = LL
//      tag (1), high 32 = n_ids (0x2000) | +0x18 n_peers = 7 row7: +0x00 coll_rx (rank 0's own
//      slice source, all zero) | +0x08 residual | +0x10 hidden_out | +0x18 epoch | +0x20/+0x28 NaN
//      fill f32/bf16
//    What the indexer kernel needs for q=1: kv_len = cur_pos + 1; sparse fork `cur_pos >= 0x7ff`,
//    top-k participation `cur_pos >= 0x800`, N_active = min((cur_pos + 0x80) >> 7, 0x80), one H-bin
//    atomic per CTA, Group2b only CTA 132, causal length cur_pos + 1, 1024 LL records (2048 ids)
//    published as {id0, 1, id1, 1} rows to every id_peers[] recv_buf, and the same K5 push/receive
//    as below with slot = rank - (rank > dst) (rank 0 -> slot 0 of every pure rank); bf16 NaN
//    refill limited to 4608 B.
// ---------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------

//    before the launch (external reset, mechanism outside the kernel), terminal values after it
//    (for q=1):
//      C0 +0x00 = 0x100 (128 Group1 CTAs x +2 after P1)          C1 +0x10 = 0x100 (128 x +2 after
//      WqB) C2 +0x20 = 0x100 (16 RoPE tails 132..147 x +0x10)          C3 +0x30 = 0x100 (4 kv-norm
//      tails 128..131 x +0x40) C4 +0x40 = 0x100 (128 x +2 after ProjQ)                     C6 +0x50
//      = 0x100 (q1: 32 attention CTAs x +8; q4: 128 x +2) C5 +0x54 = 0x80000000 (attention CTAs:
//      ctaid 0 adds kC5Head<Q>, the others +1; waiters test the sign bit) C7 +0x60 = 0x100 (128 x
//      +2 after ProjO)
//    Waiters test `>= 0x100` / sign bit (C5). Arrive = block barrier, then thread 0: fence + atomic
//    add.
// ---------------------------------------------------------------------------------------------------------------
constexpr int kWordC0 = 0x00 / 4, kWordC1 = 0x10 / 4, kWordC2 = 0x20 / 4, kWordC3 = 0x30 / 4,
              kWordC4 = 0x40 / 4;
constexpr int kWordC6 = 0x50 / 4, kWordC5 = 0x54 / 4, kWordC7 = 0x60 / 4;
constexpr uint32_t kSyncFull = 0x100u;
constexpr uint32_t kWeightGroup1 = 2u;     // C0, C1, C4, C7 (and C6 for q4)
constexpr uint32_t kWeightRope = 0x10u;    // C2
constexpr uint32_t kWeightKvNorm = 0x40u;  // C3
template <int Q>
constexpr uint32_t kWeightC6 = (Q == 1) ? 8u : 2u;
template <int Q>
constexpr uint32_t kC5Head = (Q == 1) ? 0x7fffffe1u : 0x7fffff81u;  // + (Q*32 - 1) x 1 = 0x80000000
constexpr uint32_t kSyncBlockBytes = 0x100;  // 8 rows x 16 B (the host snapshots 0x100 B)

// ---------------------------------------------------------------------------------------------------------------
// 5. Geometry shared by both instantiations.
// ---------------------------------------------------------------------------------------------------------------
constexpr int kHidden = 6144, kQlora = 2048, kKvlora = 512, kNope = 192, kRope = 64, kHeads = 10,
              kVdim = 256, kSelected = 2048;
constexpr int kGroup1Ctas = 128;
constexpr int kQkvaCtas =
    82;  // 32-row blocks: rows 0..2047 -> post_q, 2048..2559 -> post_kv, 2560..2623 -> k_pe raw
constexpr int kQkvaRows = kQlora + kKvlora + kRope;  // 2624
constexpr int kWqbCtas = 80;  // 32 packed rows each: 0..1919 = head*192 + dim (nope), 1920..2559 =
                              // 1920 + head*64 + dim (pe)
constexpr int kWqbRows = kHeads * (kNope + kRope);
constexpr int kWqbNopeRows = kHeads * kNope;  // 1920
constexpr int kProjqCtas = 80;                // 64 rows each of head*512 + out
constexpr int kProjoCtas = 80;                // 32 rows each of head*256 + out
constexpr int kUnprojCtas = 128;              // 48 rows each of 6144
constexpr int kUnprojRows = 48;
constexpr int kSplitRows = 64;
constexpr int kSplits =
    32;  // q1: attention CTA = split (0..31); q4: (query = ctaid >> 5, split = ctaid & 31)
constexpr int kKvNormTail =
    128;  // CTAs 128..131: kv-norm of token t = ctaid - 128 (t < Q works, the rest only arrive C3)
constexpr int kRopeTail =
    132;  // CTAs 132..147: RoPE of token t = ctaid - 132 (t < Q works, the rest only arrive C2)
constexpr int kLseStride = 12;  // f32 per split in O_LSE_ACC (10 live + 2 pad)
constexpr size_t kOAccSplitElems = size_t(kHeads) * kKvlora;           // 5120 f32 = 0x5000 B
constexpr size_t kOAccTokenElems = size_t(kSplits) * kOAccSplitElems;  // 0xa0000 B
constexpr size_t kLseTokenElems = size_t(kSplits) * kLseStride;        // 0x600 B
constexpr int32_t kIdSentinel = 0x32000;  // dense ids past min(kv_len - 1, 2047)
constexpr float kExp2Scale = 0.09016844f;
constexpr float kRmsEps = 1e-5f;
constexpr uint32_t kNanF32 = 0x7fc00000u;
constexpr uint32_t kNanBf16x2 = 0x7fc07fc0u;
constexpr int kNanF32Lines = 1024;  // 16 KB of 16 B lines
template <int Q>
constexpr int kNanBf16Lines =
    Q * 288;  // [q][9][256] bf16 = q x 4608 B (q1: idx <= 0x11f, q4: <= 0x47f)

// ---------------------------------------------------------------------------------------------------------------
// 6. K5 push collective
//    destination block of (token t, sender rank, receiver dst, CTA c):
//        peers[dst] + base + t*0x38000 + slot*0x8000 + parity*0x4000 + c*0x80,   slot = rank -
//        (rank > dst),   parity = epoch & 1,

//    line k = 6 bf16 (this CTA's UnProj rows 48c+6k..48c+6k+5) in dwords 0..2 + E (epoch low 32
//    bits) in dword 3. Receiver (own inbox peers[rank], slot s <- rank s + (s >= rank)) polls every
//    line until its dword 3 == E. hidden[t][48c + i] = bf16(f32(bf16(sum_{rank 0..7}
//    f32(slice_rank[i]) )) + f32(residual[t][48c + i]) ), rank order, own slice inserted at its
//    rank.  The q1 instance only touches token slice 0 (224 KB of the 896 KB inbox).
// ---------------------------------------------------------------------------------------------------------------
constexpr size_t kInboxTokenStride = 0x38000;  // 7 slots x 0x8000
constexpr size_t kInboxSlotStride = 0x8000;    // 2 parities x 0x4000
constexpr size_t kInboxParityStride = 0x4000;  // 128 CTAs x 0x80
constexpr size_t kInboxCtaStride = 0x80;       // 8 lines x 16 B
template <int Q>
constexpr size_t kInboxBytes = size_t(Q) * kInboxTokenStride;  // q1 0x38000, q4 0xe0000
constexpr int kLlRecordsPerCta =
    32;  // NCCL-LL ids rows: record r = (q*32 + split)*32 + lane, 16 B {id0, tag, id1, tag}
constexpr size_t kLlRecvInts = size_t(8) * 2048 * 2;

// Dynamic shared memory the kernel uses (bytes) per instantiation; see ModeSmem in kernel.cu.
// Exposed for the host launcher.
template <int Q>
constexpr int kModeSmemBytes =
    ((Q * kHidden * 2 > kHeads * (kKvlora + kRope) * 4) ? Q * kHidden * 2
                                                        : kHeads * (kKvlora + kRope) * 4) +
    Q * 32 * 8 * 4 + 128 + kHeads * kSplitRows * 4 + kSplitRows * 4 + 512;

}  // namespace flashinfer::glm5_tp8::mla_q1
