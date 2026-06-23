#!/usr/bin/env python3
"""Precision/bench tests for the W8A8 blockwise batched GEMM on gfx1250.

Operation (E8M0 blockwise dequant, both operands fp8):
    C[b,m,n] = sum_k (A_fp8[b,m,k] * a_scale[b,m,k//gk])
                   * (B_fp8[b,n,k] * b_scale[b,n//gn,k//gk])

Layouts (per spec):
    A       : [B, M, K]            fp8_e4m3fn   (1×128 quant granularity)
    a_scale : [B, M, K//gk]        uint8 E8M0   (one byte per A-row per k-block)
    B       : [B, N, K]            fp8_e4m3fn   (128×128 quant granularity)
    b_scale : [B, N//gn, K//gk]    uint8 E8M0   (one byte per 128-col block per k-block)
    C       : [B, M, N]            bf16

E8M0 byte ``e`` decodes to the scalar ``2**(e - 127)`` (OCP micro-exponent),
so we build scales as exact powers of two and the reference dequant is exact;
the only error vs. the kernel is bf16 output rounding (WMMA accumulates in f32).

Reference:
    a_deq = A_fp8.float() * 2**(a_scale-127) expanded over k-block
    b_deq = B_fp8.float() * 2**(b_scale-127) expanded over (n-block, k-block)
    ref   = einsum("bmk,bnk->bmn", a_deq, b_deq).bfloat16()
"""

import argparse
import os
import sys
import time

# tests/kernels/ -> FlyDSL (for kernels.* / tests.*) and project root (for the
# kernel module itself, which lives at the repo top level next to FlyDSL/).
_FLYDSL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PROJ_ROOT = os.path.dirname(_FLYDSL_ROOT)
_PYFLIR_SRC = os.path.join(_FLYDSL_ROOT, "flydsl", "src")
for _p in (_FLYDSL_ROOT, _PROJ_ROOT, _PYFLIR_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import flydsl  # noqa: E402,F401

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

from flydsl.runtime.device import get_rocm_arch
from tests.test_common import verify_output

from FlyDSL.kernels.bmm_w8a8_gfx1250 import compile_bmm_w8a8_preshuffle_gfx1250


if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _make_fp8(shape, scale=0.1):
    """Random fp8_e4m3fn tensor (CPU). Falls back to int8 if fp8 is unavailable."""
    raw = torch.randn(shape, dtype=torch.float32) * scale
    try:
        t = raw.clamp(-1.0, 1.0).to(torch.float8_e4m3fn)
        return t, t.float()
    except (AttributeError, RuntimeError):
        q = (raw.clamp(-1.0, 1.0) * 127).round().to(torch.int8)
        return q, q.float() / 127.0


def _make_e8m0(shape, lo=126, hi=130):
    """Random E8M0 scale bytes in [lo, hi] and their decoded fp32 scalars.

    Bytes stay near the 127 (==1.0) bias so the accumulation magnitude is sane;
    the decoded value is the exact power of two ``2**(byte-127)``.
    """
    bytes_u8 = torch.randint(lo, hi + 1, shape, dtype=torch.int32)
    decoded = torch.exp2((bytes_u8 - 127).float())
    return bytes_u8.to(torch.uint8), decoded


def run_w8a8_test(
    B, M, N, K,
    group_k=128, group_n=128,
    tile_m=128, tile_n=256, tile_k=128,
    num_buffers=4,
    m_warp=2, n_warp=2,
    out_dtype="bf16",
    l2_prefetch_distance=2,
    use_tdm_store=True,
    expert_sched_mode=True,
    inst_prefetch=False,
    cluster_m=1, cluster_n=1,
    waves_per_eu=None,
    wave_specialized_tdm=False,
    atol=5e-2, rtol=5e-2,
    check=True,
    bench=False, bench_iters=100,
):
    arch = str(get_rocm_arch())
    if arch != "gfx1250":
        pytest.skip(f"W8A8 BMM requires gfx1250, got {arch}")

    if N % tile_n != 0:
        pytest.skip(f"N={N} must be divisible by tile_n={tile_n}")
    if K % tile_k != 0:
        pytest.skip(f"K={K} must be divisible by tile_k={tile_k}")
    if tile_k != group_k:
        pytest.skip(f"tile_k ({tile_k}) must equal group_k ({group_k})")
    if (K // group_k) != 32:
        pytest.skip(f"kernel preloads a full 32-entry scale row, got K//gk={K // group_k}")

    mpad = _align_up(M, tile_m)
    gk, gn = group_k, group_n

    torch.manual_seed(0)

    # ---- inputs (CPU) ----
    a_fp8, a_f = _make_fp8((B, M, K))                       # [B,M,K]
    b_fp8, b_f = _make_fp8((B, N, K))                       # [B,N,K]
    a_sc_u8, a_sc = _make_e8m0((B, M, K // gk))            # [B,M,K//gk]
    b_sc_u8, b_sc = _make_e8m0((B, N // gn, K // gk))      # [B,N//gn,K//gk]

    # ---- reference (f32 dequant -> einsum -> bf16) ----
    a_sc_exp = a_sc.repeat_interleave(gk, dim=2)                              # [B,M,K]
    b_sc_exp = b_sc.repeat_interleave(gn, dim=1).repeat_interleave(gk, dim=2)  # [B,N,K]
    a_deq = a_f * a_sc_exp
    b_deq = b_f * b_sc_exp
    ref = torch.einsum("bmk,bnk->bmn", a_deq, b_deq).to(torch.bfloat16)       # [B,M,N]

    # ---- pad M (dim=1) for A / a_scale / C ----
    def _pad_m(t, mpad, fill=None):
        if mpad == t.shape[1]:
            return t
        pad = torch.zeros((t.shape[0], mpad - t.shape[1], *t.shape[2:]), dtype=t.dtype)
        if fill is not None:
            pad = pad + fill
        return torch.cat([t, pad.to(t.dtype)], dim=1)

    a_pad = _pad_m(a_fp8, mpad)
    a_sc_pad = _pad_m(a_sc_u8, mpad, fill=127)  # pad scale with 1.0 (e8m0=127)

    # ---- move to GPU ----
    a_gpu = a_pad.cuda().contiguous()
    b_gpu = b_fp8.cuda().contiguous()
    a_sc_gpu = a_sc_pad.cuda().contiguous()
    b_sc_gpu = b_sc_u8.cuda().contiguous()
    c_gpu = torch.zeros((B, mpad, N), dtype=torch.bfloat16, device="cuda")

    print(
        f"w8a8 BMM B={B} M={M}(pad={mpad}) K={K} N={N} "
        f"gk={gk} gn={gn} tile={tile_m}x{tile_n}x{tile_k} "
        f"bufs={num_buffers} m_warp={m_warp} n_warp={n_warp} "
        f"tdm_store={use_tdm_store} out_dtype={out_dtype} "
        f"expert_sched={expert_sched_mode} cluster=({cluster_m},{cluster_n}) "
        f"wpe={waves_per_eu} ws_tdm={wave_specialized_tdm}"
    )

    launch_fn = compile_bmm_w8a8_preshuffle_gfx1250(
        B=B, M=mpad, N=N, K=K,
        group_k=gk, group_n=gn,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp,
        out_dtype=out_dtype,
        num_buffers=num_buffers,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_store=use_tdm_store,
        expert_sched_mode=expert_sched_mode,
        inst_prefetch=inst_prefetch,
        cluster_m=cluster_m, cluster_n=cluster_n,
        waves_per_eu=waves_per_eu,
        wave_specialized_tdm=wave_specialized_tdm,
    )

    a_flat = a_gpu.view(-1)
    b_flat = b_gpu.view(-1)
    a_sc_flat = a_sc_gpu.view(-1)
    b_sc_flat = b_sc_gpu.view(-1)
    c_flat = c_gpu.view(-1)

    launch_fn(c_flat, a_flat, b_flat, a_sc_flat, b_sc_flat, mpad,
              torch.cuda.current_stream())
    torch.cuda.synchronize()

    if check:
        c_out = c_gpu[:, :M, :].cpu().float()
        ok = verify_output(c_out, ref.float(), rtol=rtol, atol=atol)
        if not ok:
            max_diff = (c_out - ref.float()).abs().max().item()
            print(f"  FAILED: max_diff={max_diff:.4f}")
            assert ok, f"Precision check FAILED: max_diff={max_diff:.4f}"
        print("  PASSED")

    if bench:
        for _ in range(10):
            launch_fn(c_flat, a_flat, b_flat, a_sc_flat, b_sc_flat, mpad,
                      torch.cuda.current_stream())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            launch_fn(c_flat, a_flat, b_flat, a_sc_flat, b_sc_flat, mpad,
                      torch.cuda.current_stream())
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        us = (t1 - t0) / bench_iters * 1e6
        flops = 2 * B * M * N * K
        tflops = flops / (us * 1e-6) / 1e12
        # A fp8 + B fp8 + a_scale u8 + b_scale u8 + C bf16
        mem_bytes = (B * M * K + B * N * K
                     + B * M * (K // gk) + B * (N // gn) * (K // gk)
                     + B * M * N * 2)
        bw_tbs = mem_bytes / (us * 1e-6) / 1e12
        print(f"  bench: {us:.2f} µs  {tflops:.2f} TFLOPS  {bw_tbs:.2f} TB/s")


# ---------------------------------------------------------------------------
# Tests (fixed N=1024, K=4096 per spec; tile 128x256x128, m_warp=n_warp=2, nb=4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M", [128, 256])
def test_w8a8_spec_shapes(M):
    """Spec config: B=16, N=1024, K=4096, tile 128x256x128, num_buffers=4."""
    run_w8a8_test(
        B=16, M=M, N=1024, K=4096,
        group_k=128, group_n=128,
        tile_m=128, tile_n=256, tile_k=128,
        m_warp=2, n_warp=2,
        num_buffers=4,
    )


@pytest.mark.parametrize("M", [64, 130])
def test_w8a8_m_padding(M):
    """Non-tile-multiple M exercises the mpad path and [:, :M, :] slice."""
    run_w8a8_test(
        B=16, M=M, N=1024, K=4096,
        group_k=128, group_n=128,
        tile_m=128, tile_n=256, tile_k=128,
        m_warp=2, n_warp=2,
        num_buffers=4,
    )


def test_w8a8_small_batch():
    """Smaller batch keeps total runtime down while still tiling N/K fully."""
    run_w8a8_test(
        B=4, M=256, N=1024, K=4096,
        group_k=128, group_n=128,
        tile_m=128, tile_n=256, tile_k=128,
        m_warp=2, n_warp=2,
        num_buffers=4,
    )


@pytest.mark.parametrize("use_tdm_store", [True, False])
def test_w8a8_epilogue_variants(use_tdm_store):
    """TDM store vs. buffer store epilogue must agree with the reference."""
    run_w8a8_test(
        B=4, M=128, N=1024, K=4096,
        group_k=128, group_n=128,
        tile_m=128, tile_n=256, tile_k=128,
        m_warp=2, n_warp=2,
        num_buffers=4,
        use_tdm_store=use_tdm_store,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="W8A8 blockwise BMM gfx1250 test/bench")
    parser.add_argument("-B", type=int, default=16)
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("--group-k", type=int, default=128)
    parser.add_argument("--group-n", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=2)
    parser.add_argument("--n-warp", type=int, default=2)
    parser.add_argument("--out-dtype", type=str, default="bf16")
    parser.add_argument("--num-buffers", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--no-tdm-store", action="store_true")
    parser.add_argument("--no-expert-sched-mode", action="store_true")
    parser.add_argument("--inst-prefetch", action="store_true")
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--waves-per-eu", type=int, default=None)
    parser.add_argument("--l2-prefetch-distance", type=int, default=2)
    parser.add_argument("--wave-specialized-tdm", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-iters", type=int, default=100)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_w8a8_test(
        B=args.B, M=args.M, N=args.N, K=args.K,
        group_k=args.group_k, group_n=args.group_n,
        tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
        m_warp=args.m_warp, n_warp=args.n_warp,
        out_dtype=args.out_dtype,
        num_buffers=args.num_buffers,
        use_tdm_store=not args.no_tdm_store,
        expert_sched_mode=not args.no_expert_sched_mode,
        inst_prefetch=args.inst_prefetch,
        cluster_m=args.cluster_m, cluster_n=args.cluster_n,
        waves_per_eu=args.waves_per_eu,
        wave_specialized_tdm=args.wave_specialized_tdm,
        l2_prefetch_distance=args.l2_prefetch_distance,
        check=not args.no_check,
        bench=args.bench, bench_iters=args.bench_iters,
    )
