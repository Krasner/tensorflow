/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "tensorflow/core/kernels/flash_attention/fa2_api.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op_gpu.cu.h"

namespace tensorflow::functor {

// __launch_bounds__ lets ptxas use as many registers as Th threads per block
// (and kMinBlocks blocks per SM) allow; without it, the toolchain's
// --maxrregcount=0 clamps the kernel to 24 and the per-thread accumulators
// spill to local memory.
template <typename T, int D_MAX>
__global__ void __launch_bounds__(traits::FlashTraits<D_MAX>::Th,
                                  traits::FlashTraits<D_MAX>::kMinBlocks)
    FlashAttnFwdKernel(const T *__restrict__ Q, const T *__restrict__ K,
                       const T *__restrict__ V, T *__restrict__ Out,
                       float *__restrict__ stats_out, int BH, int H, int S_q,
                       int S_kv, int D_qk, int D_v, bool causal_mask,
                       float dropout_rate, float scale, FlashAttentionRng rng,
                       bool load_chunks) {
  const uint64_t seed = ResolveSeed(rng);
  const uint64_t offset = ResolveOffset(rng);
  constexpr int Br = traits::FlashTraits<D_MAX>::Br;
  constexpr int Bc = traits::FlashTraits<D_MAX>::Bc;
  // Keys per step of the inner loop: their dot products are independent, so
  // they overlap, and the running max is updated once per step.
  constexpr int kKeys = traits::FlashTraits<D_MAX>::kKeys;
  static_assert(Bc % kKeys == 0);
  constexpr int kStages = traits::FwdStages<T, D_MAX>();
  // Each thread reads its own Q row: up to D_MAX = 64 it's read straight into
  // registers (next to acc_o), beyond that it's read from a rotated tile.
  // Every thread reads the same K/V row, so those need no rotation.
  constexpr bool kQInRegisters = D_MAX <= 64;
  static_assert(kQInRegisters == (traits::FwdQTileBytes<T, D_MAX>() == 0));
  constexpr int kShiftQ = kQInRegisters ? 0 : 1;
  using TileQ = Tile<T, D_MAX, kShiftQ>;
  using TileKV = Tile<T, D_MAX, 0>;
  const float dp_inverse =
      (dropout_rate > 0.0f) ? (1.0f / (1.0f - dropout_rate)) : 1.0f;

  // Shared Memory Layout:
  // kStages x (K_tile[Bc][D_qk] | V_tile[Bc][D_v]) | Q_tile[Br][D_qk] (only
  // without kQInRegisters)
  extern __shared__ __align__(16) char smem_[];
  T *s_KV = reinterpret_cast<T *>(smem_);
  T *s_Q = s_KV + kStages * 2 * Bc * D_MAX;

  // Grid Mapping
  // blockIdx.x covers all tiles: (Batch * Heads * S_q) / Br
  const int tiles_per_batch = (S_q + Br - 1) / Br;
  const bool block_valid =
      blockIdx.x <
      tiles_per_batch * BH;  // in case current block exceeds total blocks
  const int bh = block_valid ? blockIdx.x / tiles_per_batch : 0;
  const int b = bh / H;
  const int h = bh - b * H;
  // Stats and the dropout RNG index (batch, head) pairs as bh.
  const size_t batch_offset_q = static_cast<size_t>(bh) * S_q;
  const size_t row_base_q = RowBase(b, h, S_q, H);
  const size_t row_base_kv = RowBase(b, h, S_kv, H);

  const int tile_idx_q = blockIdx.x % tiles_per_batch;
  const int tile_offset_q = tile_idx_q * Br;
  const int tid = threadIdx.x;
  const int q_offset = tile_offset_q + tid;
  const bool active = block_valid && (tid < Br) && (q_offset < S_q);

  // Registers for Output Accumulation
  float acc_o[D_MAX];
#pragma unroll
  for (int d = 0; d < D_MAX; ++d) {
    acc_o[d] = ZEROF;
  }
  float m_i = NEGINF;
  float l_i = ZEROF;  // Keeps track of sum(exp(S - m))

  const uint64_t global_rng_idx =
      seed + offset + static_cast<uint64_t>((batch_offset_q + q_offset) * S_kv);

  float q_reg[kQInRegisters ? D_MAX : 1];
  if constexpr (kQInRegisters) {
    const T *q_row = Q + (row_base_q + static_cast<size_t>(q_offset) * H) * D_qk;
#pragma unroll
    for (int d = 0; d < D_MAX; ++d) {
      q_reg[d] = (active && d < D_qk) ? static_cast<float>(q_row[d]) : ZEROF;
    }
  } else if (block_valid) {
    // Arrives with the first K/V tile.
    LoadTile<T, D_MAX, kShiftQ>(load_chunks, s_Q, Q, tid, blockDim.x, Br, D_qk,
                                S_q, tile_offset_q, row_base_q, H);
  }

  // Causal: this Q tile's rows only attend to keys up to its last row, so KV
  // tiles starting past tile_offset_q + Br - 1 are fully masked. The bound is
  // the same for the whole block, so every thread runs the same iterations and
  // the __syncthreads() stay matched.
  const int kv_end =
      !block_valid ? 0
                   : (causal_mask ? std::min(S_kv, tile_offset_q + Br) : S_kv);
  const int kv_tiles = (kv_end + Bc - 1) / Bc;
  auto load_kv = [&](int t) {
    T *s_K = s_KV + (t % kStages) * 2 * Bc * D_MAX;
    LoadTile<T, D_MAX, 0>(load_chunks, s_K, K, tid, blockDim.x, Bc, D_qk, S_kv,
                          t * Bc, row_base_kv, H);
    LoadTile<T, D_MAX, 0>(load_chunks, s_K + Bc * D_MAX, V, tid, blockDim.x,
                          Bc, D_v, S_kv, t * Bc, row_base_kv, H);
  };

  if constexpr (kStages > 1) {
    if (kv_tiles > 0) load_kv(0);
    CpAsyncCommit();
  }
  for (int t = 0; t < kv_tiles; ++t) {
    if constexpr (kStages == 1) {
      load_kv(t);
      CpAsyncCommit();
    }
    CpAsyncWait<0>();
    // Makes tile t visible to the block. With two stages, it also means every
    // thread is done with tile t - 1, so its buffer can take tile t + 1, which
    // then loads while this tile is processed.
    __syncthreads();
    if constexpr (kStages > 1) {
      if (t + 1 < kv_tiles) load_kv(t + 1);
      CpAsyncCommit();
    }
    const T *s_K = s_KV + (t % kStages) * 2 * Bc * D_MAX;
    const T *s_V = s_K + Bc * D_MAX;
    const int tile_offset_kv = t * Bc;

    if (active) {
      int valid_end = std::min(Bc, S_kv - tile_offset_kv);
      if (causal_mask) {
        valid_end = std::min(valid_end, q_offset - tile_offset_kv + 1);
      }
      // Online softmax, kKeys keys at a time, so no scores are buffered.
      // acc_o and l_i are only rescaled when the running max grows, which
      // becomes rare after the first few keys. Keys past valid_end (rows of
      // the tile past S_kv or masked by causality) get probability 0.
      for (int j = 0; j < valid_end; j += kKeys) {
        float score[kKeys];
        if constexpr (kQInRegisters) {
          TileKV::template DotN<kKeys>(q_reg, s_K, j, score);
        } else {
          TileDotN<TileKV, TileQ, kKeys>(s_K, j, s_Q, tid, score);
        }
        float m_new = m_i;
#pragma unroll
        for (int k = 0; k < kKeys; ++k) {
          score[k] = (j + k < valid_end) ? score[k] * scale : -INFINITY;
          m_new = fmaxf(m_new, score[k]);
        }
        if (m_new > m_i) {
          const float ratio = Eigen::numext::exp(m_i - m_new);
          l_i *= ratio;
#pragma unroll
          for (int d = 0; d < D_MAX; ++d) {
            acc_o[d] *= ratio;
          }
          m_i = m_new;
        }

        float p[kKeys];
#pragma unroll
        for (int k = 0; k < kKeys; ++k) {
          p[k] = Eigen::numext::exp(score[k] - m_i);
          l_i += p[k];
          if (dropout_rate > 0.0f) {
            const int kv_offset = tile_offset_kv + j + k;
            if (get_random_float_v2(global_rng_idx, kv_offset) <
                dropout_rate) {
              p[k] = ZEROF;
            } else {
              p[k] *= dp_inverse;
            }
          }
        }
        TileKV::template AxpyN<kKeys>(acc_o, p, s_V, j);
      }
    }
    if constexpr (kStages == 1) {
      // The next iteration overwrites this tile.
      __syncthreads();
    }
  }

  if (active) {
    // l_i >= 1 once any key is attended (the max term contributes exp(0)), so
    // the LSE is exact. A row without keys gets -inf, the log-sum-exp of the
    // empty set, as for S_kv == 0.
    const bool has_keys = l_i > ZEROF;
    stats_out[batch_offset_q + q_offset] =
        has_keys ? m_i + Eigen::numext::log(l_i) : -INFINITY;
    float inv_l = has_keys ? (1.0f / l_i) : ZEROF;
    const size_t out_base =
        (row_base_q + static_cast<size_t>(q_offset) * H) * D_v;
#pragma unroll
    for (int d = 0; d < D_MAX; ++d) {
      if (d < D_v) {
        Out[out_base + d] = static_cast<T>(acc_o[d] * inv_l);
      }
    }
  }
}

// The float version of FlashAttnFwdKernel, written as matrix products so that
// each value read from shared memory feeds several multiply-adds. A block owns
// Br = 64 query rows; its 256 threads form a 16 x 16 grid (ty, tx). For each
// KV tile of Bc keys:
//  1. S = Q K^T: thread (ty, tx) computes rows ty + 16a and keys tx + 16b.
//  2. Online softmax: a row's scores are spread over the 16 threads with the
//     same ty (one half of a warp), which reduce the running max with
//     shuffles. Each keeps a partial row sum, rescaled with the row's factor,
//     and adds them up at the end. P goes to shared memory.
//  3. O += P V: thread (ty, tx) owns rows ty + 16a and columns
//     [4 (tx + 16e), +4).
template <int D_MAX>
__global__ void __launch_bounds__(traits::FwdGemmLayout<D_MAX>::Th,
                                  traits::FwdGemmLayout<D_MAX>::kMinBlocks)
    FlashAttnFwdGemmKernel(const float *__restrict__ Q,
                           const float *__restrict__ K,
                           const float *__restrict__ V, float *__restrict__ Out,
                           float *__restrict__ stats_out, int BH, int H,
                           int S_q, int S_kv, int D_qk, int D_v,
                           bool causal_mask, float dropout_rate, float scale,
                           FlashAttentionRng rng) {
  using L = traits::FwdGemmLayout<D_MAX>;
  constexpr int Br = L::Br;
  constexpr int Bc = L::Bc;
  constexpr int kStages = L::kStages;
  constexpr int kLd = L::kLd;
  constexpr int kLdP = L::kLdP;
  constexpr int kRA = Br / 16;     // rows per thread
  constexpr int kKB = Bc / 16;     // keys per thread in step 1
  constexpr int kDC = D_MAX / 64;  // 4-column chunks per thread in step 3
  static_assert(L::Th == 256 && Bc % 16 == 0 && D_MAX % 64 == 0);

  const uint64_t seed = ResolveSeed(rng);
  const uint64_t offset = ResolveOffset(rng);
  const float dp_inverse =
      (dropout_rate > 0.0f) ? (1.0f / (1.0f - dropout_rate)) : 1.0f;

  // Shared Memory Layout:
  // Q[Br][kLd] | kStages x (K[Bc][kLd] | V[Bc][kLd]) | P[Br][kLdP]
  extern __shared__ __align__(16) char smem_[];
  float *s_Q = reinterpret_cast<float *>(smem_);
  float *s_stages = s_Q + Br * kLd;
  float *s_P = s_stages + kStages * L::kStageFloats;

  const int tid = threadIdx.x;
  const int tx = tid % 16;
  const int ty = tid / 16;

  // Blocks go through the Q tiles last to first (across all heads), so that
  // with causal masking the tiles with the most keys start first and the
  // short ones fill in at the end.
  const int tiles_per_batch = (S_q + Br - 1) / Br;
  const bool block_valid = blockIdx.x < tiles_per_batch * BH;
  const int bh = block_valid ? blockIdx.x % BH : 0;
  const int tile_idx_q =
      block_valid ? tiles_per_batch - 1 - static_cast<int>(blockIdx.x / BH)
                  : 0;
  const int b = bh / H;
  const int h = bh - b * H;
  const int tile_offset_q = tile_idx_q * Br;
  const size_t batch_offset_q = static_cast<size_t>(bh) * S_q;
  const size_t row_base_q = RowBase(b, h, S_q, H);
  const size_t row_base_kv = RowBase(b, h, S_kv, H);

  float acc_o[kRA][4 * kDC];
  float m_i[kRA], l_i[kRA];  // l_i: this thread's keys only
#pragma unroll
  for (int ra = 0; ra < kRA; ++ra) {
    m_i[ra] = NEGINF;
    l_i[ra] = ZEROF;
#pragma unroll
    for (int d = 0; d < 4 * kDC; ++d) acc_o[ra][d] = ZEROF;
  }

  // Arrives with the first K/V tile.
  if (block_valid) {
    LoadRowsPadded<D_MAX, kLd>(s_Q, Q, tid, L::Th, Br, D_qk, S_q,
                               tile_offset_q, row_base_q, H);
  }

  // As in FlashAttnFwdKernel: with causal masking, KV tiles past the Q tile's
  // last row are skipped; the range is the same for the whole block.
  const int kv_end =
      !block_valid ? 0
                   : (causal_mask ? std::min(S_kv, tile_offset_q + Br) : S_kv);
  const int kv_tiles = (kv_end + Bc - 1) / Bc;
  auto stage = [&](int t) {
    return s_stages + (t % kStages) * L::kStageFloats;
  };
  auto load_kv = [&](int t) {
    float *sK = stage(t);
    LoadRowsPadded<D_MAX, kLd>(sK, K, tid, L::Th, Bc, D_qk, S_kv, t * Bc,
                               row_base_kv, H);
    LoadRowsPadded<D_MAX, kLd>(sK + Bc * kLd, V, tid, L::Th, Bc, D_v, S_kv,
                               t * Bc, row_base_kv, H);
  };

  if constexpr (kStages > 1) {
    if (kv_tiles > 0) load_kv(0);
    CpAsyncCommit();
  }
  for (int t = 0; t < kv_tiles; ++t) {
    if constexpr (kStages == 1) {
      load_kv(t);
      CpAsyncCommit();
    }
    CpAsyncWait<0>();
    // Makes tile t visible. Every thread is also done with tile t - 1, so its
    // stage and the P tile can be overwritten.
    __syncthreads();
    if constexpr (kStages > 1) {
      if (t + 1 < kv_tiles) load_kv(t + 1);
      CpAsyncCommit();
    }
    const float *sK = stage(t);
    const float *sV = sK + Bc * kLd;
    const int tile_offset_kv = t * Bc;

    // 1. S = Q K^T.
    float acc_s[kRA][kKB];
#pragma unroll
    for (int ra = 0; ra < kRA; ++ra) {
#pragma unroll
      for (int kb = 0; kb < kKB; ++kb) acc_s[ra][kb] = ZEROF;
    }
#pragma unroll 4
    for (int k = 0; k < D_MAX; k += 4) {
      float4 q[kRA], kk[kKB];
#pragma unroll
      for (int ra = 0; ra < kRA; ++ra) {
        q[ra] =
            *reinterpret_cast<const float4 *>(s_Q + (ty + 16 * ra) * kLd + k);
      }
#pragma unroll
      for (int kb = 0; kb < kKB; ++kb) {
        kk[kb] =
            *reinterpret_cast<const float4 *>(sK + (tx + 16 * kb) * kLd + k);
      }
#pragma unroll
      for (int ra = 0; ra < kRA; ++ra) {
#pragma unroll
        for (int kb = 0; kb < kKB; ++kb) {
          acc_s[ra][kb] = Dot4(q[ra], kk[kb], acc_s[ra][kb]);
        }
      }
    }

    // 2. Online softmax. Keys past S_kv and causally masked ones get -inf,
    // so probability 0.
#pragma unroll
    for (int ra = 0; ra < kRA; ++ra) {
      const int r = ty + 16 * ra;
      const int q_offset = tile_offset_q + r;
      float m_tile = -INFINITY;
#pragma unroll
      for (int kb = 0; kb < kKB; ++kb) {
        const int kv_offset = tile_offset_kv + tx + 16 * kb;
        const bool valid =
            kv_offset < S_kv && !(causal_mask && q_offset < kv_offset);
        acc_s[ra][kb] = valid ? acc_s[ra][kb] * scale : -INFINITY;
        m_tile = fmaxf(m_tile, acc_s[ra][kb]);
      }
#pragma unroll
      for (int lane_mask = 1; lane_mask < 16; lane_mask <<= 1) {
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, lane_mask));
      }
      const float m_new = fmaxf(m_i[ra], m_tile);
      const float ratio = Eigen::numext::exp(m_i[ra] - m_new);
      m_i[ra] = m_new;
      l_i[ra] *= ratio;
#pragma unroll
      for (int d = 0; d < 4 * kDC; ++d) acc_o[ra][d] *= ratio;
      const uint64_t global_rng_idx =
          seed + offset +
          static_cast<uint64_t>((batch_offset_q + q_offset) * S_kv);
#pragma unroll
      for (int kb = 0; kb < kKB; ++kb) {
        const int c = tx + 16 * kb;
        float p_ij = Eigen::numext::exp(acc_s[ra][kb] - m_new);
        l_i[ra] += p_ij;
        if (dropout_rate > 0.0f) {
          if (get_random_float_v2(global_rng_idx, tile_offset_kv + c) <
              dropout_rate) {
            p_ij = ZEROF;
          } else {
            p_ij *= dp_inverse;
          }
        }
        s_P[r * kLdP + c] = p_ij;
      }
    }
    __syncthreads();

    // 3. O += P V.
#pragma unroll 2
    for (int c = 0; c < Bc; c += 4) {
      float4 p[kRA];
#pragma unroll
      for (int ra = 0; ra < kRA; ++ra) {
        p[ra] =
            *reinterpret_cast<const float4 *>(s_P + (ty + 16 * ra) * kLdP + c);
      }
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
#pragma unroll
        for (int e = 0; e < kDC; ++e) {
          const float4 v = *reinterpret_cast<const float4 *>(
              sV + (c + cc) * kLd + 4 * (tx + 16 * e));
#pragma unroll
          for (int ra = 0; ra < kRA; ++ra) {
            const float pv = Lane(p[ra], cc);
            acc_o[ra][4 * e + 0] += pv * v.x;
            acc_o[ra][4 * e + 1] += pv * v.y;
            acc_o[ra][4 * e + 2] += pv * v.z;
            acc_o[ra][4 * e + 3] += pv * v.w;
          }
        }
      }
    }
    if constexpr (kStages == 1) {
      // The next iteration overwrites this K/V tile.
      __syncthreads();
    }
  }
  // The Q copies are still in flight if there was no KV tile.
  CpAsyncWait<0>();

#pragma unroll
  for (int ra = 0; ra < kRA; ++ra) {
    float l = l_i[ra];
#pragma unroll
    for (int lane_mask = 1; lane_mask < 16; lane_mask <<= 1) {
      l += __shfl_xor_sync(0xffffffffu, l, lane_mask);
    }
    const int q_offset = tile_offset_q + ty + 16 * ra;
    if (!block_valid || q_offset >= S_q) continue;
    // As in FlashAttnFwdKernel: l >= 1 once any key is attended, and a row
    // without keys gets an LSE of -inf and a zero output.
    const bool has_keys = l > ZEROF;
    if (tx == 0) {
      stats_out[batch_offset_q + q_offset] =
          has_keys ? m_i[ra] + Eigen::numext::log(l) : -INFINITY;
    }
    const float inv_l = has_keys ? (1.0f / l) : ZEROF;
    float *out = Out + (row_base_q + static_cast<size_t>(q_offset) * H) * D_v;
#pragma unroll
    for (int e = 0; e < kDC; ++e) {
      const int col = 4 * (tx + 16 * e);
      if (col >= D_v) continue;  // D_v is a multiple of 4
      *reinterpret_cast<float4 *>(out + col) = make_float4(
          acc_o[ra][4 * e] * inv_l, acc_o[ra][4 * e + 1] * inv_l,
          acc_o[ra][4 * e + 2] * inv_l, acc_o[ra][4 * e + 3] * inv_l);
    }
  }
}

template <typename T>
struct FlashAttentionFunctor<GPUDevice, T> {
  static constexpr bool kIsBf16 = std::is_same_v<T, Eigen::bfloat16>;
  static constexpr bool kIsFa2Type = kIsBf16 || std::is_same_v<T, Eigen::half>;

  static absl::Status CheckFeatureSize(int D_qk, int D_v) {
    if (std::max(D_qk, D_v) <= 128) return absl::OkStatus();
    if constexpr (kIsFa2Type) {
      if (flash_attention_fa2::Supported(D_qk, D_v)) return absl::OkStatus();
    }
    return FeatureSizeError(D_qk, D_v);
  }

  absl::Status operator()(const GPUDevice &d,
                          typename TTypes<T, 4>::ConstTensor Q,
                          typename TTypes<T, 4>::ConstTensor K,
                          typename TTypes<T, 4>::ConstTensor V,
                          typename TTypes<T, 4>::Tensor Out,
                          typename TTypes<float, 3>::Tensor Stats,
                          bool causal_mask, float dropout_rate, float scale,
                          FlashAttentionRng rng) const {
    const int B = Q.dimension(0);
    const int S_q = Q.dimension(1);
    const int H = Q.dimension(2);
    const int D_qk = Q.dimension(3);
    const int S_kv = K.dimension(1);
    const int D_v = V.dimension(3);
    int max_D = std::max(D_qk, D_v);

    if (B == 0 || H == 0 || S_q == 0) return absl::OkStatus();
    if (S_kv == 0) {
      // No keys to attend to: the output is zero, and the log-sum-exp of the
      // empty set is -inf.
      TF_RETURN_IF_ERROR(
          ZeroOutputs(d, {{Out.data(), Out.size() * sizeof(T)}}));
      return FillFloats(d, Stats.data(), Stats.size(), -INFINITY);
    }

    if constexpr (kIsFa2Type) {
      if (flash_attention_fa2::Supported(D_qk, D_v) &&
          flash_attention_fa2::GridSupported(B, H)) {
        const cudaError_t err = flash_attention_fa2::Forward(
            d.stream(), kIsBf16, Q.data(), K.data(), V.data(), Out.data(),
            Stats.data(), B, H, S_q, S_kv, D_qk, scale, dropout_rate,
            causal_mask, rng.seed, rng.offset, rng.seed_dev, rng.offset_dev);
        if (err != cudaSuccess) {
          return errors::Internal("FlashAttention-2 forward failed: ",
                                  cudaGetErrorString(err));
        }
        return absl::OkStatus();
      }
    }

    if (max_D <= 8) {
      return LaunchKernel<8>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk, D_v,
                             causal_mask, dropout_rate, scale, rng);
    } else if (max_D <= 16) {
      return LaunchKernel<16>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk,
                              D_v, causal_mask, dropout_rate, scale, rng);
    } else if (max_D <= 32) {
      return LaunchKernel<32>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk,
                              D_v, causal_mask, dropout_rate, scale, rng);
    } else if (max_D <= 64) {
      return LaunchKernel<64>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk,
                              D_v, causal_mask, dropout_rate, scale, rng);
    } else if (max_D <= 128) {
      return LaunchKernel<128>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk,
                               D_v, causal_mask, dropout_rate, scale, rng);
    } else {
      return FeatureSizeError(D_qk, D_v);
    }
  }

  template <int D_MAX>
  absl::Status LaunchGemmKernel(const GPUDevice &d, const float *Q,
                                const float *K, const float *V, float *Out,
                                float *Stats, int BH, int H, int S_q, int S_kv,
                                int D_qk, int D_v, bool causal_mask,
                                float dropout_rate, float scale,
                                FlashAttentionRng rng) const {
    using Layout = traits::FwdGemmLayout<D_MAX>;
    constexpr int smem_size = Layout::SmemBytes();
    auto kernel = FlashAttnFwdGemmKernel<D_MAX>;
    if (cudaError_t err = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        err != cudaSuccess) {
      return errors::Internal("FlashAttention: setting shared memory failed: ",
                              cudaGetErrorString(err));
    }
    const uint64_t total_blocks =
        static_cast<uint64_t>(BH) * ((S_q + Layout::Br - 1) / Layout::Br);
    if (total_blocks > 2147483647ULL) {
      return errors::InvalidArgument(
          "FlashAttention: total blocks exceed 1D grid limit");
    }
    return GpuLaunchKernel(kernel, static_cast<int>(total_blocks), Layout::Th,
                           smem_size, d.stream(), Q, K, V, Out, Stats, BH, H,
                           S_q, S_kv, D_qk, D_v, causal_mask, dropout_rate,
                           scale, rng);
  }

  template <int D_MAX>
  absl::Status LaunchKernel(const GPUDevice &d,
                            typename TTypes<T, 4>::ConstTensor Q,
                            typename TTypes<T, 4>::ConstTensor K,
                            typename TTypes<T, 4>::ConstTensor V,
                            typename TTypes<T, 4>::Tensor Out,
                            typename TTypes<float, 3>::Tensor Stats, int B,
                            int H, int S_q, int S_kv, int D_qk, int D_v,
                            bool causal_mask, float dropout_rate, float scale,
                            FlashAttentionRng rng) const {
    const int BH = B * H;  // fits in int, see CheckDimsFitInt32
    const bool load_chunks =
        CanLoadChunks<T>({Q.data(), K.data(), V.data()}, {D_qk, D_v});
    if constexpr (std::is_same_v<T, float> && D_MAX >= 64) {
      if (load_chunks &&
          MaxSmemOptin() >= traits::FwdGemmLayout<D_MAX>::SmemBytes()) {
        return LaunchGemmKernel<D_MAX>(d, Q.data(), K.data(), V.data(),
                                       Out.data(), Stats.data(), BH, H, S_q,
                                       S_kv, D_qk, D_v, causal_mask,
                                       dropout_rate, scale, rng);
      }
    }
    constexpr int Br = traits::FlashTraits<D_MAX>::Br;
    constexpr int threads_per_block = traits::FlashTraits<D_MAX>::Th;
    constexpr int smem_size = traits::FwdSmemBytes<T, D_MAX>();
    uint64_t blocks_per_batch = (S_q + Br - 1) / Br;
    uint64_t total_blocks = static_cast<uint64_t>(BH) * blocks_per_batch;
    if (total_blocks > 2147483647ULL) {
      return errors::InvalidArgument(
          "FlashAttention: total blocks exceed 1D grid limit");
    }
    return GpuLaunchKernel(FlashAttnFwdKernel<T, D_MAX>, total_blocks,
                           threads_per_block, smem_size, d.stream(), Q.data(),
                           K.data(), V.data(), Out.data(), Stats.data(), BH, H,
                           S_q, S_kv, D_qk, D_v, causal_mask, dropout_rate,
                           scale, rng, load_chunks);
  }
};

template struct FlashAttentionFunctor<GPUDevice, float>;
template struct FlashAttentionFunctor<GPUDevice, Eigen::half>;
template struct FlashAttentionFunctor<GPUDevice, Eigen::bfloat16>;

}  // namespace tensorflow::functor

#endif
