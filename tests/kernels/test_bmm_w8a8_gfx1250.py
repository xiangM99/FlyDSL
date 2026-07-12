#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Batched FP8 (A8W8) blockscale GEMM test for gfx1250.

Exercises ``compile_blockscale_w8a8_bmm`` from
``kernels.gemm.bmm_w8a8_gfx1250`` for correctness and performance.

All tensors carry a leading batch dim B. The kernel is layout-agnostic: per-batch
offsets are driven by runtime batch strides, so both ``[B, M, K]`` (batch-outermost)
and ``[M, B, K]`` (batch-interleaved) layouts are covered by picking the matching
per-tensor batch strides.

Quantization: FP8/E4M3 activation + weight, one E8M0 (uint8) scale per 128-wide
K block (and per 128-wide N block for B), applied in-MMA via V_WMMA_SCALE.
"""

import logging
import os
import sys

import pytest
import torch
import torch.nn.functional as F

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_REPO_ROOT, "build", "python_packages"),
    _REPO_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.bmm_w8a8_gfx1250 import compile_blockscale_w8a8_bmm  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402
from tests.utils import get_dtype_max, shuffle_weight  # noqa: E402

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = get_rocm_arch()
if not str(ARCH).startswith("gfx1250"):
    pytest.skip(
        f"bmm_w8a8 blockscale tests require gfx1250, got {ARCH}",
        allow_module_level=True,
    )

DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in str(ARCH) else torch.float8_e4m3fnuz

SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128


# ─────────────────────────── quantization helpers ───────────────────────────
def _per_block_fp8_quant_lastdim(x: torch.Tensor, block: int = SCALE_BLOCK_K):
    """Quantize a [..., K] fp32 tensor to raw FP8 bytes + per-`block`-K E8M0 scale.

    Returns (y_uint8 [..., K], scale_uint8 [..., K//block]).
    """
    if x.shape[-1] % block != 0:
        raise ValueError(f"Last dim must be divisible by {block}, got {x.shape[-1]}")
    shape_original = x.shape
    x2d = x.reshape(-1, shape_original[-1]).to(torch.float32)
    rows, cols = x2d.shape
    x_blk = x2d.reshape(-1, block)
    x_blk = torch.nan_to_num(x_blk, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = torch.amax(torch.abs(x_blk), dim=1)
    dtype_max = float(get_dtype_max(DTYPE_FP8))
    scale_e8m0 = fp4_utils.f32_to_e8m0(max_abs / dtype_max)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)
    scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
    scale_f32[scale_f32 == 0] = 1.0
    y_f32 = x_blk / scale_f32.view(-1, 1)
    y_f32 = torch.clamp(y_f32, min=-dtype_max, max=dtype_max)
    y = fp4_utils._f32_to_floatx_unpacked(y_f32.contiguous().view(-1), 4, 3).view(torch.uint8)
    y = y.view(*shape_original)
    scale = scale_e8m0.view(rows, cols // block).view(torch.uint8)
    return y, scale.view(*shape_original[:-1], cols // block)


def _dequant_fp8(x_q_u8: torch.Tensor, scale_u8: torch.Tensor, block: int = SCALE_BLOCK_K):
    """Dequantize raw FP8 bytes + per-block E8M0 scale back to fp32."""
    scale_f32 = fp4_utils.e8m0_to_f32(scale_u8.view(torch.uint8))
    scale_expanded = scale_f32.repeat_interleave(block, dim=-1)[..., : x_q_u8.shape[-1]]
    return fp4_utils.fp8_e4m3_to_f32(x_q_u8.view(torch.uint8)) * scale_expanded


def _quant_b_blockscale(w: torch.Tensor):
    """Quantize weight [B, N, K] fp32 with per-(128N x 128K) block E8M0 scale.

    Returns (w_q_u8 [B, N, K], w_scale_u8 [B, N//128, K//128]).
    """
    Bb, N, K = w.shape
    assert N % SCALE_BLOCK_N == 0 and K % SCALE_BLOCK_K == 0
    w32 = w.to(torch.float32)
    # Reshape into (B, N//bn, bn, K//bk, bk) and reduce each 2-D block to one scale.
    blk = w32.view(Bb, N // SCALE_BLOCK_N, SCALE_BLOCK_N, K // SCALE_BLOCK_K, SCALE_BLOCK_K)
    max_abs = blk.abs().amax(dim=(2, 4))  # [B, N//bn, K//bk]
    dtype_max = float(get_dtype_max(DTYPE_FP8))
    scale_e8m0 = fp4_utils.f32_to_e8m0(max_abs / dtype_max)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)
    scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
    scale_f32[scale_f32 == 0] = 1.0
    # Broadcast scale back to element granularity for quantization.
    s_full = scale_f32.unsqueeze(2).unsqueeze(4)  # [B, N//bn, 1, K//bk, 1]
    y = (blk / s_full)
    y = torch.clamp(y, min=-dtype_max, max=dtype_max)
    y = fp4_utils._f32_to_floatx_unpacked(y.contiguous().view(-1), 4, 3).view(torch.uint8)
    y = y.view(Bb, N, K)
    return y, scale_e8m0.view(torch.uint8).view(Bb, N // SCALE_BLOCK_N, K // SCALE_BLOCK_K)


# ─────────────────────────── torch reference ───────────────────────────
def run_torch_bmm_blockscale(a_q, a_scale, b_q, b_scale, out_dtype):
    """Reference: dequantize A/B and do a batched matmul C = A @ B^T.

    a_q:     [B, M, K] uint8 fp8
    a_scale: [B, M, K//128] uint8 e8m0
    b_q:     [B, N, K] uint8 fp8
    b_scale: [B, N//128, K//128] uint8 e8m0
    Returns C [B, M, N] fp32.
    """
    Bb, M, K = a_q.shape
    N = b_q.shape[1]
    a_f32 = _dequant_fp8(a_q, a_scale, SCALE_BLOCK_K)  # [B, M, K]
    # Expand B scale to [B, N, K] element granularity.
    bs_f32 = fp4_utils.e8m0_to_f32(b_scale.view(torch.uint8))  # [B, N//128, K//128]
    bs_full = (
        bs_f32.repeat_interleave(SCALE_BLOCK_N, dim=1)[:, :N, :]
        .repeat_interleave(SCALE_BLOCK_K, dim=2)[:, :, :K]
    )
    b_f32 = fp4_utils.fp8_e4m3_to_f32(b_q.view(torch.uint8)) * bs_full  # [B, N, K]
    out = torch.bmm(a_f32, b_f32.transpose(1, 2))  # [B, M, N]
    torch_dt = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    return out.to(torch_dt).to(torch.float32)


# ─────────────────────────── tile config ───────────────────────────
def select_tile_config(M, N, K):
    """A small heuristic to pick a valid tile for the given shape."""
    candidates = [
        (128, 128, 128),
        (256, 256, 128),
        (128, 256, 128),
        (256, 128, 128),
    ]

    def _valid(tm, tn, tk):
        return N % tn == 0 and K % tk == 0 and (K // tk) >= 2

    valid = [c for c in candidates if _valid(*c)]
    if not valid:
        return (128, 128, 128)
    # Prefer the deep-pipeline eligible 256x256 when M is large enough.
    if M >= 256 and (256, 256, 128) in valid:
        return (256, 256, 128)
    return valid[0]


# ─────────────────────────── the test ───────────────────────────
@pytest.mark.parametrize(
    "B, M, N, K",
    [
        pytest.param(2, 128, 512, 512, id="B2-128x512x512"),
        pytest.param(4, 256, 512, 1024, id="B4-256x512x1024"),
        pytest.param(8, 128, 1024, 512, id="B8-128x1024x512"),
        pytest.param(2, 512, 1024, 2048, id="B2-512x1024x2048", marks=pytest.mark.large_shape),
        pytest.param(3, 129, 512, 512, id="B3-129x512x512-partialM"),
    ],
)
@pytest.mark.parametrize("batch_layout", ["bmk", "mbk"])
@pytest.mark.parametrize("out_dtype", ["bf16", "f16"])
def test_bmm_w8a8_blockscale(
    B,
    M,
    N,
    K,
    batch_layout,
    out_dtype,
    *,
    bench_iters=20,
    bench_warmup=3,
    test_graph=False,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    tile_m, tile_n, tile_k = select_tile_config(M, N, K)

    print("=" * 80)
    print(
        f"BMM W8A8 blockscale: B={B}, M={M}, N={N}, K={K}, "
        f"layout={batch_layout}, tile=({tile_m}x{tile_n}x{tile_k}), out={out_dtype}"
    )
    print("=" * 80)

    scale_k = K // SCALE_BLOCK_K
    scale_n = N // SCALE_BLOCK_N

    # ── build random fp32 inputs, then quantize ──
    a_f = (torch.rand((B, M, K), dtype=torch.float32, device=device) - 0.5) / 5
    b_f = (torch.rand((B, N, K), dtype=torch.float32, device=device) - 0.5) / 5

    a_q, a_scale = _per_block_fp8_quant_lastdim(a_f, SCALE_BLOCK_K)  # [B,M,K],[B,M,scale_k]
    b_q, b_scale = _quant_b_blockscale(b_f)  # [B,N,K],[B,scale_n,scale_k]

    # ── reference ──
    c_ref = run_torch_bmm_blockscale(a_q, a_scale, b_q, b_scale, out_dtype)  # [B,M,N] f32

    # ── kernel-side tensor layouts ──
    # A data as fp8 element type. Batch layout selects the physical arrangement.
    a_fp8 = a_q.view(DTYPE_FP8)  # [B, M, K]
    # B is preshuffled per-batch with the (16,16) WMMA layout the kernel expects.
    b_shuffled = torch.stack(
        [shuffle_weight(b_q[i], layout=(16, 16)) for i in range(B)], dim=0
    ).contiguous()  # [B, N//16, K*16] (uint8), reinterpreted below
    b_fp8 = b_shuffled.view(DTYPE_FP8)

    # A-scale: col-major preferred → transpose K-block dim to be outer, M inner.
    #   layout [B, scale_k, M] uint8, unit-stride along M.
    a_scale_cm = a_scale.transpose(1, 2).contiguous()  # [B, scale_k, M]
    # B-scale: row-major [B, scale_n, scale_k] uint8, unit-stride along scale_k.
    b_scale_rm = b_scale.contiguous()  # [B, scale_n, scale_k]

    # ── choose physical batch layout & per-tensor batch strides ──
    if batch_layout == "bmk":
        a_dev = a_fp8.contiguous()
        c_dev = torch.zeros((B, M, N), dtype=torch_out_dtype, device=device)
        as_dev = a_scale_cm.contiguous()  # [B, scale_k, M]
        stride_a_batch = M * K
        stride_c_batch = M * N
        stride_ascale_batch = scale_k * M
    else:  # "mbk": batch interleaved between M and K
        a_dev = a_fp8.transpose(0, 1).contiguous()  # [M, B, K]
        c_dev = torch.zeros((M, B, N), dtype=torch_out_dtype, device=device)
        as_dev = a_scale_cm.transpose(1, 2).transpose(0, 1).contiguous()  # [M, scale_k, B]? -> keep [scale_k, B, M]
        # For col-major A-scale the inner (unit) dim is M; put batch just outside it.
        as_dev = a_scale_cm.permute(1, 0, 2).contiguous()  # [scale_k, B, M]
        stride_a_batch = K            # advancing one batch = one K row in [M,B,K]
        stride_c_batch = N            # [M,B,N]
        stride_ascale_batch = M       # [scale_k, B, M]

    # B and B-scale keep [B, ...] outermost in both cases (weights are per-batch).
    b_dev = b_fp8.contiguous()
    bs_dev = b_scale_rm.contiguous()
    stride_b_batch = (N // 16) * (K * 16)  # elements in one preshuffled batch slab
    stride_bscale_batch = scale_n * scale_k

    # leading dims (row strides) — dense: lda = K, ldc = N.
    lda = K
    ldc = N
    # A-scale col-major: stride between adjacent K-blocks is M (unit stride is M).
    stride_ascale_k = M
    stride_ascale_m = scale_k  # unused in col-major path but must be a sane value

    # ── compile ──
    exe = compile_blockscale_w8a8_bmm(
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_dtype=out_dtype,
        ascale_layout="col_major",
        scale_block_k=SCALE_BLOCK_K,
        scale_block_n=SCALE_BLOCK_N,
    )
    print("  Compiled OK")

    stream = torch.cuda.current_stream()

    def _args(c, a, b, sa, sb):
        return (
            c, a, b, sa, sb,
            B, M, N, lda, ldc,
            stride_ascale_m, stride_ascale_k,
            stride_a_batch, stride_b_batch, stride_c_batch,
            stride_ascale_batch, stride_bscale_batch,
            stream,
        )

    compiled_exe = flyc.compile(exe, *_args(c_dev, a_dev, b_dev, as_dev, bs_dev))

    def launch_kernel(c, a, b, sa, sb):
        compiled_exe(*_args(c, a, b, sa, sb))

    launch_kernel(c_dev, a_dev, b_dev, as_dev, bs_dev)
    torch.cuda.synchronize()

    # ── gather output back to [B, M, N] for comparison ──
    if batch_layout == "bmk":
        c_out = c_dev
    else:
        c_out = c_dev.transpose(0, 1).contiguous()  # [M,B,N] -> [B,M,N]
    c_out_f32 = c_out.to(torch.float32)

    passed = verify_output(c_out_f32, c_ref, rtol=1e-2, atol=0.05)

    # ── performance ──
    bench_iters = max(2, int(bench_iters))
    _, us = run_perftest(
        launch_kernel,
        c_dev,
        a_dev,
        b_dev,
        as_dev,
        bs_dev,
        num_iters=bench_iters,
        num_warmup=int(bench_warmup),
        testGraph=test_graph,
    )
    torch.cuda.synchronize()

    flops = 2 * B * M * N * K
    elem_bytes = 1  # fp8
    bytes_moved = (
        B * M * K * elem_bytes
        + B * N * K * elem_bytes
        + B * M * N * 2
        + B * (M * scale_k + scale_n * scale_k)  # e8m0 scale bytes
    )
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"  Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")

    assert passed, "Kernel output verification failed"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BMM W8A8 blockscale benchmark")
    parser.add_argument("-B", type=int, default=4)
    parser.add_argument("-M", type=int, default=256)
    parser.add_argument("-N", type=int, default=512)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("--tile_m", type=int, default=None)
    parser.add_argument("--tile_n", type=int, default=None)
    parser.add_argument("--tile_k", type=int, default=None)
    parser.add_argument("--layout", type=str, default="bmk", choices=["bmk", "mbk"])
    parser.add_argument("--out_dtype", type=str, default="bf16", choices=["f16", "bf16"])
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--graph", action="store_true", default=False)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    test_bmm_w8a8_blockscale(
        B=args.B,
        M=args.M,
        N=args.N,
        K=args.K,
        batch_layout=args.layout,
        out_dtype=args.out_dtype,
        bench_iters=args.num_iters,
        bench_warmup=args.num_warmup,
        test_graph=args.graph,
    )
