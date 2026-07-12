"""FP8 blockscale batched GEMM (BMM) kernel for gfx1250.

FP8 (E4M3) activation and weight with E8M0 block scales applied in-MMA via
V_WMMA_SCALE at 128-K/128-N granularity

All tensors are 3-D with a leading batch dimension B. Per-batch memory offsets
are driven entirely by runtime batch strides (passed as kernel arguments), so
either ``[B, M, K]`` (batch-outermost) or ``[M, B, K]`` (batch-interleaved)
layouts are supported without recompilation — the host picks the layout by
passing the matching per-tensor batch stride. The batch index rides grid.z,
folded with split_k (``grid.z = B * split_k``).
"""

import functools
import warnings

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    idx2crd,
    range_constexpr,
    rocdl,
)
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from flydsl.expr import arith, buffer_ops, const_expr, gpu, idx2crd, range_constexpr, rocdl, tdm_ops
from kernels.gemm.gemm_common_gfx1250 import (
    extract_lds_base_idx,
    get_lds_memref,
    lds_load_b32_raw,
    lds_load_b128_raw,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    store_acc_vec8_to_buffer,
    store_acc_vec8_to_lds,
)
from kernels.mma.pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes


def _s_prefetch_inst_burst(num_pages: int, page_bytes: int = 4096):
    """gfx1250: prefetch ``num_pages`` × 4 KB of instructions ahead of PC.

    Caller must keep ``num_pages * page_bytes`` within shader bounds; over-reach
    page-faults.
    """
    from flydsl._mlir.dialects import llvm as _llvm

    lines = [
        f"s_prefetch_inst_pc_rel {pg * page_bytes}, null, 31" for pg in range(num_pages)
    ]
    _llvm.inline_asm(None, [], "\n".join(lines), "", has_side_effects=True)


# Common constants
WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32


def _align_up(value: int, align: int) -> int:
    if value % align == 0:
        return value
    return (value + align - 1) // align * align


LDS_PAD_A_BYTES = 16
LDS_PAD_D_BYTES = 16
LDS_SEGMENT_BYTES = 64 * 1024
LDS_GFX1250_MAX_BYTES = 5 * LDS_SEGMENT_BYTES


@functools.lru_cache(maxsize=256)
def compile_blockscale_w8a8_bmm(
    *,
    N: int = 0,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    num_buffers: int = 2,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    cluster_m: int = 1,
    cluster_n: int = 1,
    out_dtype: str = "bf16",
    inst_prefetch: bool = False,
    split_k: int = 1,
    expert_sched_mode: bool = True,
    atomic_barrier_enable: bool = False,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    ascale_layout: str = "row_major",
):
    """Compile an FP8 blockscale batched GEMM kernel with TDM async copy.

    FP8 (E4M3) activation and weight with E8M0 block scales applied in-MMA via
    V_WMMA_SCALE at 128-K/128-N granularity. All tensors carry a leading batch
    dimension B; per-batch offsets come from runtime batch strides so both
    ``[B, M, K]`` and ``[M, B, K]``-style layouts work without recompilation.

    Returns a JitFunction:
        launch_fn(arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale,
                  M, N, lda, ldc, stride_ascale_m, stride_ascale_k,
                  batch, stride_a_batch, stride_b_batch, stride_c_batch,
                  stride_ascale_batch, stride_bscale_batch, stream)
    """
    if scale_block_k != WMMA_K or scale_block_n != 128:
        raise ValueError(
            "blockscale requires scale_block_k=128 and scale_block_n=128"
        )
    if ascale_layout not in ("row_major", "col_major"):
        raise ValueError(
            f"ascale_layout must be 'row_major' or 'col_major', got {ascale_layout!r}"
        )
    if ascale_layout == "row_major":
        warnings.warn(
            "blockscale ascale_layout='row_major' has a 1-byte-strided A-scale TDM "
            "and may be slow; prefer ascale_layout='col_major'.",
            stacklevel=2,
        )

    if out_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"out_dtype must be 'bf16', or 'f16', got {out_dtype!r}"
        )
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4
    effective_expert_sched_mode = bool(expert_sched_mode)

    if num_buffers not in (2, 3, 4, 5, 6):
        raise ValueError(f"num_buffers must be 2, 3, 4, 5 or 6, got {num_buffers}")
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    tdm_store_enabled = split_k == 1

    use_cluster = cluster_m > 1 or cluster_n > 1
    if use_cluster:
        if cluster_m * cluster_n > 16:
            raise ValueError(
                f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}"
            )
        if (N // tile_n) % cluster_n != 0:
            raise ValueError(
                f"cluster_n={cluster_n} must divide N/tile_n={N // tile_n} "
                f"(N={N}, tile_n={tile_n}): gfx1250 does not support partial clusters"
            )
    effective_waves_per_eu = waves_per_eu

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE
    if block_threads > 1024:
        raise ValueError(f"block_threads must be <= 1024, got {block_threads}")

    # ── FP8 compile-time constants ──
    WMMA_N_EFF = 16  # N-cols covered per WMMA instruction
    ACC_VEC_SIZE = 8  # accumulator vector width
    DS_LOADS_PER_A_FRAG = 4

    packed_tile_k_a = tile_k  # FP8: 1 byte per element
    packed_tile_k_b = tile_k
    K_packed_a = K
    K_packed_b = K
    K_blockscale = K // scale_block_k
    split_k_chunk = K // split_k

    if K % tile_k != 0:
        raise ValueError(f"K must be divisible by tile_k={tile_k}, got K={K}")
    if K % split_k != 0:
        raise ValueError(f"K must be divisible by split_k={split_k}, got K={K}")
    if split_k_chunk % tile_k != 0:
        raise ValueError(
            f"K/split_k must be divisible by tile_k={tile_k}, got {split_k_chunk}"
        )
    if tile_k % WMMA_K != 0:
        raise ValueError(f"tile_k must be a multiple of {WMMA_K}, got {tile_k}")
    if tile_m % WMMA_M != 0:
        raise ValueError(f"tile_m must be a multiple of {WMMA_M}, got {tile_m}")
    if tile_n % WMMA_N != 0:
        raise ValueError(f"tile_n must be a multiple of {WMMA_N}, got {tile_n}")
    if packed_tile_k_a % 4 != 0:
        raise ValueError(
            f"packed_tile_k_a must be a multiple of 4, got {packed_tile_k_a}"
        )
    if packed_tile_k_b % 4 != 0:
        raise ValueError(
            f"packed_tile_k_b must be a multiple of 4, got {packed_tile_k_b}"
        )

    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N_EFF != 0:
        raise ValueError(
            f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N_EFF}"
        )

    if K % scale_block_k != 0 or N % scale_block_n != 0:
        raise ValueError(
            f"blockscale requires K%{scale_block_k}==0 and N%{scale_block_n}==0; got K={K}, N={N}"
        )

    num_k_tiles = split_k_chunk // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, got {num_k_tiles}"
        )

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N_EFF
    n_accs = wmma_m_rep * wmma_n_rep

    b_scale_load_rep = wmma_n_rep

    # Blockscale A/B-scale layout: [M, K//128] / [N//128, K//128] uint8 E8M0.
    ascale_col_major = ascale_layout == "col_major"
    bsc_a_row_stride_bytes = k_wmma_steps
    bsc_b_row_stride_bytes = k_wmma_steps
    bsc_b_tile_blocks = max(
        (bn + tile_n - 1) // scale_block_n - bn // scale_block_n + 1
        for bn in range(0, N, tile_n)
    )
    if ascale_col_major:
        # LDS holds [k_wmma_steps][tile_m]: M contiguous, K strided by tile_m.
        lds_as_row_stride = 1
        lds_as_ks_stride = tile_m
    else:
        # LDS holds [tile_m][k_wmma_steps]: K contiguous, M strided by k_wmma_steps.
        lds_as_row_stride = k_wmma_steps
        lds_as_ks_stride = 1

    # M-half op_sel pairing: lane_kgrp selects the upper/lower half of the warp's
    # M span.
    ascale_opsel = wmma_m_rep >= 2 and (wmma_m_rep & (wmma_m_rep - 1)) == 0
    ascale_half = wmma_m_rep // 2
    ascale_load = ascale_half if ascale_opsel else wmma_m_rep

    # TDM loader assignment: wave0=A, wave1=B, wave2=A-scale, wave3=B-scale.
    # blockscale's raw scale layout isn't preshuffled, so both scales are TDM
    # tensors. With 2/3 waves the missing scale descriptor rides as a secondary
    # issue.
    two_wave_scale = num_warps == 2
    three_wave_bscale = num_warps == 3
    secondary_scale_tdm = two_wave_scale or three_wave_bscale

    if num_warps < 2:
        raise ValueError(
            f"wave-specialized TDM requires at least 2 waves, got {num_warps}"
        )

    _b_frag_loads_per_wn = 4
    _a_frag_loads_per_wm = 4
    # blockscale scales are delivered outside the LDS ds_read path (via VGPR
    # prefetch), so no scale ds_loads ride the streaming schedule.
    _scale_ds_loads = 0
    _a_frag_ds = wmma_m_rep * _a_frag_loads_per_wm
    _bs_ds_loads = wmma_n_rep * _b_frag_loads_per_wn + _scale_ds_loads
    _as_ds_loads = _a_frag_ds + _scale_ds_loads
    _row_major_k_prefetch_bundle_ds = _a_frag_ds + _bs_ds_loads

    _a_pad_dwords = packed_tile_k_a // 4
    _a_pad_pow2 = _a_pad_dwords > 0 and (_a_pad_dwords & (_a_pad_dwords - 1)) == 0
    lds_pad_a_bytes = LDS_PAD_A_BYTES if _a_pad_pow2 else 0
    lds_a_stride_bytes = packed_tile_k_a + lds_pad_a_bytes

    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = tile_n * packed_tile_k_b
    _scale_guard_bytes = 16
    # The guard tail lets compute-side scale reads safely over-read a row's bytes.
    lds_a_scale_bytes = tile_m * bsc_a_row_stride_bytes + _scale_guard_bytes
    lds_b_scale_bytes = bsc_b_tile_blocks * bsc_b_row_stride_bytes + _scale_guard_bytes

    tdm_desc_num_warps = 1

    stage_layout = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="blockscale_fp8_layout"
    )
    stage_a_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_data_rel_off + lds_a_data_bytes
    stage_b_data_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_data_rel_off + lds_b_data_bytes
    stage_a_scale_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_a_scale_rel_off + lds_a_scale_bytes
    stage_b_scale_rel_off = stage_layout._align(stage_layout.ptr, 16)
    stage_layout.ptr = stage_b_scale_rel_off + lds_b_scale_bytes
    stage_bytes = _align_up(stage_layout.ptr, 128)

    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    _base_tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)

    _last_compute_stage = _base_tail_plan[-1][1]

    stage_pitch_bytes = _align_up(stage_bytes, 1024)
    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"blockscale_fp8_{tile_m}x{tile_n}x{tile_k}_{m_warp}x{n_warp}_{num_buffers}buf_arena"
        ),
    )

    stage_phys_order = [i for i in range(num_buffers) if i != _last_compute_stage]
    stage_phys_order.append(_last_compute_stage)
    stage_base_off = [0] * num_buffers
    for phys_i, logical_i in enumerate(stage_phys_order):
        stage_base_off[logical_i] = phys_i * stage_pitch_bytes
    arena_alloc.ptr = stage_pitch_bytes * num_buffers
    arena_total_bytes = arena_alloc.ptr
    epilogue_fence_threshold_bytes = tdm_epilogue_fence_threshold_bytes(
        stage_base_off=stage_base_off,
        tail_plan=_base_tail_plan,
        loop_iters=loop_iters,
        extra=extra,
    )

    stage_a_data_off = [
        stage_base_off[i] + stage_a_data_rel_off for i in range(num_buffers)
    ]
    stage_b_data_off = [
        stage_base_off[i] + stage_b_data_rel_off for i in range(num_buffers)
    ]
    stage_a_scale_off = [
        stage_base_off[i] + stage_a_scale_rel_off for i in range(num_buffers)
    ]
    stage_b_scale_off = [
        stage_base_off[i] + stage_b_scale_rel_off for i in range(num_buffers)
    ]

    if tdm_store_enabled:
        lds_d_row_stride = warp_tile_n * elem_bytes_d + LDS_PAD_D_BYTES
        warp_d_bytes = warp_tile_m * lds_d_row_stride
        total_d_bytes = num_warps * warp_d_bytes
        d_output_off = 0
        _lds_d_stride_elems = lds_d_row_stride // 2
        _warp_d_elems = warp_d_bytes // 2
        _n_col_d_elems = WMMA_N * elem_bytes_d // 2
        d_need_epilogue_fence = total_d_bytes > epilogue_fence_threshold_bytes
        if total_d_bytes > arena_total_bytes:
            arena_total_bytes = total_d_bytes
            arena_alloc.ptr = total_d_bytes
    check_smem_capacity(arena_total_bytes, gpu_arch)

    # TENSORcnt is tracked per-wave in hardware. Keep the fence budget in stage units;
    # secondary scale descriptors on 2/3-wave paths only make this more conservative.
    TDM_LOADS_PER_STEP = 1
    tail_plan = [
        (ls, cs, o * TDM_LOADS_PER_STEP // 2 if o > 0 else o)
        for ls, cs, o in _base_tail_plan
    ]

    # Pre-compute epilogue sub-tile layout (FP8 vec8: single 8-element block).
    _sub_tiles = []
    for _wm in range(wmma_m_rep):
        for _wn in range(wmma_n_rep):
            acc_idx = _wm * wmma_n_rep + _wn
            m_off = _wm * WMMA_M
            n_sub = _wn
            _sub_tiles.append((acc_idx, 0, m_off, n_sub))

    COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING = "row_major_streaming"
    COMPUTE_SCHEDULE_FP8_QUADRANT = "fp8_quadrant"
    COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE = "fp8_deep_pipeline"

    fp8_deep_pipeline_eligible = (
        tile_m == 256
        and tile_n == 256
        and tile_k == 128
        and m_warp == 2
        and n_warp == 2
        and num_buffers == 4
        and out_dtype == "bf16"
    )

    def _pick_compute_schedule_kind():
        if wmma_m_rep % 2 != 0 or wmma_n_rep % 2 != 0 or n_accs < 8:
            return COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
        if fp8_deep_pipeline_eligible:
            return COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE
        return COMPUTE_SCHEDULE_FP8_QUADRANT

    compute_schedule_kind = _pick_compute_schedule_kind()
    use_row_major_streaming_schedule = (
        compute_schedule_kind == COMPUTE_SCHEDULE_ROW_MAJOR_STREAMING
    )
    use_fp8_quadrant_schedule = compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT
    use_fp8_deep_pipeline_schedule = (
        compute_schedule_kind == COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE
    )
    use_row_major_k_prefetch = wmma_m_rep == 1 and k_wmma_steps > 1
    _row_major_k_prefetch_depth = 2 if use_row_major_k_prefetch else 1
    _row_major_k_prefetch_depth = max(
        0, min(k_wmma_steps - 1, _row_major_k_prefetch_depth)
    )

    use_ws_tdm_split_signal_overlap = (
        (use_fp8_quadrant_schedule or use_fp8_deep_pipeline_schedule)
        and num_buffers == 4
        and use_cluster
    )
    use_tdm_late_signal_overlap = (
        use_ws_tdm_split_signal_overlap or use_row_major_k_prefetch
    )

    if use_fp8_quadrant_schedule or use_fp8_deep_pipeline_schedule:
        _fp8_half_wm = wmma_m_rep // 2
        _fp8_half_wn = wmma_n_rep // 2
        _fp8_group_size = _fp8_half_wm * _fp8_half_wn
        _fp8_b_scale_loads = 0
    if use_fp8_deep_pipeline_schedule:
        _fp8_pair_wm = 2
        _fp8_pair_wn = 2
        _fp8_wm_pairs = wmma_m_rep // _fp8_pair_wm
        _fp8_wn_pairs = wmma_n_rep // _fp8_pair_wn
        _fp8_pair_a_loads = _fp8_pair_wm * DS_LOADS_PER_A_FRAG
        _fp8_pair_b_loads = _fp8_pair_wn * _b_frag_loads_per_wn
        _fp8_scale_loads = 0

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel_blockscale_w8a8_bmm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_batch: fx.Int32,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
        i32_stride_ascale_m: fx.Int32,
        i32_stride_ascale_k: fx.Int32,
        i32_stride_a_batch: fx.Int32,
        i32_stride_b_batch: fx.Int32,
        i32_stride_c_batch: fx.Int32,
        i32_stride_ascale_batch: fx.Int32,
        i32_stride_bscale_batch: fx.Int32,
    ):
        # Enable back-to-back WMMA issue (SCHED_MODE bit[4] = DISABLE_VALU_STALL)
        rocdl.disable_xdl_arb_stall()

        if const_expr(inst_prefetch):
            if rocdl.wave_id() == fx.Int32(0):
                _s_prefetch_inst_burst(num_pages=4)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        # grid.z carries batch_idx * split_k + ks_split_idx (batch folded with split_k).
        bzz = fx.Index(gpu.block_id("z"))
        if const_expr(split_k > 1):
            batch_idx = bzz // arith.index(split_k)
            ks_split_idx = bzz % arith.index(split_k)
            split_k_base = ks_split_idx * arith.index(split_k_chunk)
        else:
            batch_idx = bzz
            split_k_base = arith.index(0)

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        # a [m,b,k] or [b,m,k]
        a_batch_off = batch_idx * fx.Index(i32_stride_a_batch)
        b_batch_off = batch_idx * fx.Index(i32_stride_b_batch)
        c_batch_off = batch_idx * fx.Index(i32_stride_c_batch)
        as_batch_off = batch_idx * fx.Index(i32_stride_ascale_batch)
        bs_batch_off = batch_idx * fx.Index(i32_stride_bscale_batch)

        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(
                local_x, local_y, cluster_m, cluster_n
            )
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0

        if const_expr(use_fp8_deep_pipeline_schedule):
            layout_thr = fx.make_layout(
                (m_warp, n_warp, 2, 16), (WAVE_SIZE, m_warp * WAVE_SIZE, 16, 1)
            )
        else:
            layout_thr = fx.make_layout(
                (m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1)
            )
        thr_coord = idx2crd(fx.Int32(tx), layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0),
            fx.get(thr_coord, 1),
            fx.get(thr_coord, 2),
            fx.get(thr_coord, 3),
        )

        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        m_idx = fx.Index(i32_m)

        _scale_identity_i32 = arith.constant(0x7F7F7F7F, type=T.i32)
        _vs_tile_a = k_wmma_steps * ascale_load

        if const_expr(True):
            _bsc_a_row0 = warp_m_base + lane16
            if const_expr(ascale_opsel):
                _bsc_a_row0 = _bsc_a_row0 + lane_kgrp * arith.index(
                    ascale_half * WMMA_M
                )

            def _broadcast_byte_i32(word, shift):
                """Replicate byte at *shift* of *word* into all 4 byte lanes."""
                byte_val = (word >> fx.Int32(shift)) if const_expr(shift != 0) else word
                return ((byte_val & fx.Int32(0xFF)) * fx.Int32(0x01010101)).ir_value()

            def _load_scale_row_bytes(lds_buf, byte_off, n):
                """Read n consecutive E8M0 scale bytes from LDS, broadcast each into a wmma_scale-ready i32. Handles any n (not just <=4)."""
                words = []
                off = byte_off
                bytes_needed = n
                while const_expr(bytes_needed > 0):
                    if const_expr(bytes_needed > 4):
                        raw = fx.Vector(lds_load_b128_raw(lds_buf, off))
                        words.extend(raw[i] for i in range(4))
                        off = off + arith.index(16)
                        bytes_needed -= 16
                    else:
                        words.append(fx.Int32(lds_load_b32_raw(lds_buf, off)))
                        off = off + arith.index(4)
                        bytes_needed -= 4
                return [
                    _broadcast_byte_i32(words[ks // 4], (ks % 4) * 8) for ks in range(n)
                ]

            _bsc_a_abs_row0 = blk_m + _bsc_a_row0  # absolute M row for OOB masking

            def load_ascale_bsc_all(lds_buf):
                vals = [None] * _vs_tile_a
                for wm in range_constexpr(ascale_load):
                    row = _bsc_a_row0 + arith.index(wm * WMMA_M)
                    if const_expr(ascale_col_major):
                        # LDS [k_wmma_steps][tile_m]
                        abs_row = _bsc_a_abs_row0 + arith.index(wm * WMMA_M)
                        row_ok = abs_row < m_idx
                        for ks in range_constexpr(k_wmma_steps):
                            off = arith.index(
                                ks * lds_as_ks_stride
                            ) + row * arith.index(lds_as_row_stride)
                            #i32
                            word = fx.Int32(lds_load_b32_raw(lds_buf, off))
                            bval = _broadcast_byte_i32(word, 0)
                            bval = arith.select(row_ok, bval, _scale_identity_i32)
                            vals[ks * ascale_load + wm] = bval
                    else:
                        byte_off = row * arith.index(lds_as_row_stride)
                        bvals = _load_scale_row_bytes(lds_buf, byte_off, k_wmma_steps)
                        for ks in range_constexpr(k_wmma_steps):
                            vals[ks * ascale_load + wm] = bvals[ks]
                return vals

            def load_bscale_bsc_all(lds_buf):
                vals = [None] * (k_wmma_steps * wmma_n_rep)
                b_wmmas_per_scale = scale_block_n // WMMA_N_EFF

                def _load_bscale_block(n_block):
                    byte_off = n_block * arith.index(bsc_b_row_stride_bytes)
                    return _load_scale_row_bytes(lds_buf, byte_off, k_wmma_steps)
                
                if const_expr(
                    tile_n % scale_block_n == 0 and scale_block_n % warp_tile_n == 0
                ):
                    n_block = warp_n_base // arith.index(scale_block_n)
                    ks_vals = _load_bscale_block(n_block)
                    for wn in range_constexpr(wmma_n_rep):
                        for ks in range_constexpr(k_wmma_steps):
                            vals[ks * wmma_n_rep + wn] = ks_vals[ks]
                    return vals

                if const_expr(
                    tile_n % scale_block_n == 0 and warp_tile_n % scale_block_n == 0
                ):
                    n_block0 = warp_n_base // arith.index(scale_block_n)
                    for nb in range_constexpr(warp_tile_n // scale_block_n):
                        ks_vals = _load_bscale_block(n_block0 + arith.index(nb))
                        for local_wn in range_constexpr(b_wmmas_per_scale):
                            wn = nb * b_wmmas_per_scale + local_wn
                            for ks in range_constexpr(k_wmma_steps):
                                vals[ks * wmma_n_rep + wn] = ks_vals[ks]
                    return vals

                _bsc_b_block_off = blk_n // arith.index(scale_block_n)
                for wn in range_constexpr(wmma_n_rep):
                    n_col = blk_n + warp_n_base + arith.index(wn * WMMA_N_EFF)
                    n_block = n_col // arith.index(scale_block_n) - _bsc_b_block_off
                    ks_vals = _load_bscale_block(n_block)
                    for ks in range_constexpr(k_wmma_steps):
                        vals[ks * wmma_n_rep + wn] = ks_vals[ks]
                return vals

        # Runtime leading-dim strides (strided A/C). Dense callers pass lda == K,
        # ldc == N for byte-identical addressing. FP8 A stride is in bytes (1 B/elem).
        lda_packed = fx.Index(i32_lda)

        stride_ascale_m = fx.Index(i32_stride_ascale_m)
        stride_ascale_k = fx.Index(i32_stride_ascale_k)

        n_stride = fx.Index(i32_ldc)
        c_nrec = fx.Index(i32_batch) * m_idx * n_stride * arith.index(elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)
        c_global_ptr_type = ir.Type.parse("!llvm.ptr<1>")
        c_global_base_i64 = llvm.PtrToIntOp(
            T.i64,
            fly.extract_aligned_pointer_as_index(
                c_global_ptr_type, arg_c.__extract_to_ir_values__()[0]
            ),
        ).result

        def make_desc_a(memref, k_base):
            k_packed_off = k_base + a_batch_off
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_a,
                lds_memref=memref,
                global_offset=(blk_m, k_packed_off),
                tensor_shape=(tile_m, packed_tile_k_a),
                strides=(lda_packed, 1),
                tile_shape=(tile_m, packed_tile_k_a),
                elem_bytes=1,
                pad_interval=packed_tile_k_a if lds_pad_a_bytes else 0,
                pad_amount=lds_pad_a_bytes,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
                oob_outer_bound=i32_m,
            )

        def make_desc_b(memref, k_base):
            k_packed_off = k_base
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b,
                lds_memref=memref,
                global_offset=(
                    blk_n // arith.index(16),
                    k_packed_off * arith.index(16) + b_batch_off,
                ),
                tensor_shape=(N // 16, K_packed_b * 16),
                strides=(K_packed_b * 16, 1),
                tile_shape=(tile_n // 16, packed_tile_k_b * 16),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )

        def make_desc_bs(memref, k_base):
            block_off = blk_n // arith.index(scale_block_n)
            col_off = k_base // arith.index(scale_block_k) + bs_batch_off
            return tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_b_scale,
                lds_memref=memref,
                global_offset=(block_off, col_off),
                tensor_shape=(N // scale_block_n, K_blockscale),
                strides=(K_blockscale, 1),
                tile_shape=(bsc_b_tile_blocks, k_wmma_steps),
                elem_bytes=1,
                pad_interval=0,
                pad_amount=0,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
                oob_outer_bound=N // scale_block_n,
            )

        def make_desc_as(memref, k_base):
            if const_expr(True):
                col_off = k_base // arith.index(scale_block_k)
                if const_expr(ascale_col_major):
                    # col_major: inner (unit-stride) dim is M — add batch there.
                    return tdm_ops.make_tensor_descriptor_2d(
                        global_ptr=arg_a_scale,
                        lds_memref=memref,
                        global_offset=(col_off, blk_m + as_batch_off),
                        tensor_shape=(K_blockscale, i32_m),
                        strides=(stride_ascale_k, 1),
                        tile_shape=(k_wmma_steps, tile_m),
                        elem_bytes=1,
                        pad_interval=0,
                        pad_amount=0,
                        num_warps=tdm_desc_num_warps,
                        workgroup_mask=a_mcast_mask,
                        atomic_barrier_enable=atomic_barrier_enable,
                        early_timeout=True,
                        oob_outer_bound=K_blockscale,
                        oob_inner_bound=as_batch_off + i32_m,
                    )
                # row_major: inner (unit-stride) dim is K//128 — add batch there.
                return tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_a_scale,
                    lds_memref=memref,
                    global_offset=(blk_m, col_off + as_batch_off),
                    tensor_shape=(tile_m, K_blockscale),
                    strides=(stride_ascale_m, 1),
                    tile_shape=(tile_m, k_wmma_steps),
                    elem_bytes=1,
                    pad_interval=0,
                    pad_amount=0,
                    num_warps=tdm_desc_num_warps,
                    workgroup_mask=a_mcast_mask,
                    atomic_barrier_enable=atomic_barrier_enable,
                    early_timeout=True,
                    oob_outer_bound=i32_m,
                )

        tdm_wave_id = rocdl.wave_id()
        tdm_wave_is_a = tdm_wave_id == fx.Int32(0)
        tdm_wave_is_b = tdm_wave_id == fx.Int32(1)
        tdm_wave_is_as = tdm_wave_id == fx.Int32(2)

        def _select_wave_tdm_value(a_value, b_value, as_value, bs_value):
            result = arith.select(tdm_wave_is_as, as_value, bs_value)
            result = arith.select(tdm_wave_is_b, b_value, result)
            return arith.select(tdm_wave_is_a, a_value, result)

        elem_ty_lds = T.f16

        def _precompute_a_lane_bases(lds_ptr):
            """Precompute per-wm A fragment lane base addresses (byte offsets)."""
            row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
            # K-dimension interleaving: kgrp0/kgrp1 read alternating 128-bit chunks
            # All formats: kgrp offset = 16 bytes (one ds_load_b128 width)
            k_half_off = lane_kgrp * arith.index(16)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                base = (
                    row_base
                    + arith.index(wm * WMMA_M * lds_a_stride_bytes)
                    + k_half_off
                )
                bases.append(base)
            return lds_ptr, bases

        def load_a_frag(lds_buffer, a_lane_base, ks):
            """Load one A-fragment from LDS.

            FP8: vec<16xi32> via 4 × ds_load_b128 (64 bytes per lane).
              Interleaved K layout:
              kgrp0 reads bytes [0:15],[32:47],[64:79],[96:111] (stride=32)
              kgrp1 reads bytes [16:31],[48:63],[80:95],[112:127] (stride=32)
            """
            k_byte_off = arith.index(ks * WMMA_K)
            byte_off = a_lane_base + k_byte_off
            v0 = fx.Vector(lds_load_b128_raw(lds_buffer, byte_off))
            # Interleaved stride=32: +0, +32, +64, +96
            v1 = fx.Vector(lds_load_b128_raw(lds_buffer, byte_off + arith.index(32)))
            v2 = fx.Vector(lds_load_b128_raw(lds_buffer, byte_off + arith.index(64)))
            v3 = fx.Vector(lds_load_b128_raw(lds_buffer, byte_off + arith.index(96)))
            v01 = v0.shuffle(v1, list(range(8)))
            v23 = v2.shuffle(v3, list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def _precompute_b_lane_bases(lds_ptr):
            """Precompute per-wn B fragment lane base addresses (byte offsets).

            FP8: 1 base per wn (16-col WMMA = 1 N-group).

            K-dimension interleaving:
              kgrp0 and kgrp1 read alternating 16x16 tiles (stride = 2 tiles).
              kgrp offset = 1 tile = 256 bytes.
            """
            _ngroup_stride = packed_tile_k_b * 16
            _n_group_base = arith.index(warp_tile_n // 16) * wave_n_idx
            row_off = lane16 * arith.index(16)
            # Interleaved — kgrp offset = 1 tile = 256 bytes
            k_tile_off = lane_kgrp * arith.index(256)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                ngroup_off = _n_group_base * arith.index(_ngroup_stride) + arith.index(
                    wn * _ngroup_stride
                )
                bases.append(ngroup_off + row_off + k_tile_off)
            return lds_ptr, bases

        def load_b_frag(lds_buffer, b_lane_bases, wn, ks):
            """Load one B-fragment from preshuffled LDS.

            FP8: 16x128 → vec<16xi32> from 1 N-group (bases[wn]).

            K-dimension interleaving:
              Stride = 2 tiles = 512 bytes between loads.
              kgrp0 reads tiles 0,2,4,6; kgrp1 reads tiles 1,3,5,7.
            """
            if const_expr(True):
                # FP8: 8 tiles per N-group
                # Interleaved stride=512: kgrp0→tiles 0,2,4,6; kgrp1→tiles 1,3,5,7
                _num_tiles = WMMA_K // 16  # 8 tiles total
                k_subtile_off = arith.index(ks * _num_tiles * 256)
                base0 = b_lane_bases[wn] + k_subtile_off
                v0 = fx.Vector(lds_load_b128_raw(lds_buffer, base0))
                v1 = fx.Vector(lds_load_b128_raw(lds_buffer, base0 + arith.index(512)))
                v2 = fx.Vector(lds_load_b128_raw(lds_buffer, base0 + arith.index(1024)))
                v3 = fx.Vector(lds_load_b128_raw(lds_buffer, base0 + arith.index(1536)))
                v01 = v0.shuffle(v1, list(range(8)))
                v23 = v2.shuffle(v3, list(range(8)))
                return v01.shuffle(v23, list(range(16)))

        # Current tile's A/B block scales (prefetched into VGPRs from LDS),
        # ordered [k_wmma_step][M-rep] / [k_wmma_step][N-rep].
        _vgpr_scale_box = [None]
        _blockscale_b_scale_box = [None]

        def _set_vgpr_a_scales(lds_as=None):
            _vgpr_scale_box[0] = load_ascale_bsc_all(lds_as)

        def _set_blockscale_b_scales(lds_bs=None):
            _blockscale_b_scale_box[0] = load_bscale_bsc_all(lds_bs)
            rocdl.s_wait_dscnt(0)

        def _load_a_scale_vgpr(ks):
            pf_a = _vgpr_scale_box[0]
            return pf_a[ks * ascale_load : (ks + 1) * ascale_load]

        def _load_b_scale_blockscale(ks):
            pf_b = _blockscale_b_scale_box[0]
            return pf_b[ks * wmma_n_rep : (ks + 1) * wmma_n_rep]

        def _load_a_scale_operand(as_buf, as_bases, ks):
            return _load_a_scale_vgpr(ks)

        def _scales_for_emit(as_buf, as_bases, bs_buf, bs_bases, ks):
            """Load scale operands for K-subtile *ks*."""
            a = _load_a_scale_operand(as_buf, as_bases, ks)
            return a, _load_b_scale_blockscale(ks)

        def _load_b_and_scales(b_buf, b_bases, as_buf, as_bases, bs_buf, bs_bases, ks):
            b_frags = [
                load_b_frag(b_buf, b_bases, wn, ks)
                for wn in range_constexpr(wmma_n_rep)
            ]
            a_scales, b_scales = _scales_for_emit(
                as_buf, as_bases, bs_buf, bs_bases, ks
            )
            return b_frags, b_scales, a_scales

        def _emit_wmma(accs, wm, wn, a_frag, b_frag, a_scales, b_scales):
            """Emit one FP8 16x16x128 blockscale WMMA instruction."""
            idx = wm * wmma_n_rep + wn
            if const_expr(ascale_opsel):
                # blockscale pairs M-blocks across the two lane_kgrp halves.
                a_scale_idx = wm % ascale_half
                a_opsel = wm // ascale_half
            else:
                a_scale_idx = wm
                a_opsel = 0

            # 16x16x128 FP8 WMMA. B-scale is one per-128-N-block scale per WMMA.
            accs[idx] = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                T.vec(8, T.f32),
                b_frag,
                a_frag,
                accs[idx],
                b_scales[wn],
                a_scales[a_scale_idx],
                fmtA=0,
                fmtB=0,
                scaleAType=0,
                scaleBType=a_opsel,
            )

        def _a_streaming_compute(
            accs,
            a_buf,
            a_bases,
            b_frags,
            b_scales,
            a_scales,
            ks,
            emit_filler=None,
            next_bs_info=None,
            mid_compute_callback=None,
        ):
            """Half-based A-streaming with zigzag wn ordering.

            When *next_bs_info* is provided, the next K-subtile's B+scale
            loads are issued BEFORE the s_wait_dscnt so they overlap with
            the current WMMA execution (partial drain pattern).
            """
            next_result = None
            _front_wm = (wmma_m_rep + 1) // 2
            _back_wm = wmma_m_rep - _front_wm

            def _emit_rows(start_wm, a_frags):
                for frag_i in range_constexpr(len(a_frags)):
                    wm = start_wm + frag_i
                    is_last = wm == wmma_m_rep - 1
                    if const_expr(is_last and emit_filler is not None):
                        rocdl.sched_barrier(0)
                        emit_filler()
                    for wn_raw in range_constexpr(wmma_n_rep):
                        wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                        _emit_wmma(
                            accs,
                            wm,
                            wn,
                            a_frags[frag_i],
                            b_frags[wn],
                            a_scales,
                            b_scales,
                        )

            a_frags_front = [
                load_a_frag(a_buf, a_bases[wm], ks) for wm in range_constexpr(_front_wm)
            ]

            _use_partial_drain = (
                next_bs_info is not None and _front_wm * wmma_n_rep >= 4
            )

            if const_expr(_use_partial_drain):
                nb_buf, nb_bases, nas_buf, nas_bases, nbs_buf, nbs_bases, n_ks = (
                    next_bs_info
                )
                next_result = _load_b_and_scales(
                    nb_buf, nb_bases, nas_buf, nas_bases, nbs_buf, nbs_bases, n_ks
                )
                rocdl.s_wait_dscnt(_bs_ds_loads)
            else:
                rocdl.s_wait_dscnt(0)

            _emit_rows(0, a_frags_front)

            if const_expr(mid_compute_callback is not None):
                rocdl.sched_barrier(0)
                mid_compute_callback()

            if const_expr(_back_wm > 0):
                a_frags_back = [
                    load_a_frag(a_buf, a_bases[_front_wm + h], ks)
                    for h in range_constexpr(_back_wm)
                ]
                _back_drain = _bs_ds_loads if _use_partial_drain else 0
                rocdl.s_wait_dscnt(_back_drain)
                _emit_rows(_front_wm, a_frags_back)

            if const_expr(_use_partial_drain):
                return accs, next_result
            if const_expr(next_bs_info is not None):
                nb_buf, nb_bases, nas_buf, nas_bases, nbs_buf, nbs_bases, n_ks = (
                    next_bs_info
                )
                next_result = _load_b_and_scales(
                    nb_buf, nb_bases, nas_buf, nas_bases, nbs_buf, nbs_bases, n_ks
                )
                return accs, next_result
            return accs

        # ── Compute on one LDS buffer ──
        def compute_tile(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
        ):
            current_accs = list(accs_in)
            _set_vgpr_a_scales(lds_as=lds_as)
            _set_blockscale_b_scales(lds_bs=lds_bs)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            # blockscale delivers scales via the VGPR box
            # (_set_vgpr_a_scales/_set_blockscale_b_scales above) — bases unused.
            as_buf, as_bases = lds_as, None
            bs_buf, bs_bases = lds_bs, None

            if const_expr(k_wmma_steps == 1):
                b_frags, b_scales, a_scales = _load_b_and_scales(
                    b_buf, b_bases, as_buf, as_bases, bs_buf, bs_bases, 0
                )
                current_accs = _a_streaming_compute(
                    current_accs,
                    a_buf,
                    a_bases,
                    b_frags,
                    b_scales,
                    a_scales,
                    0,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                )
            else:
                if const_expr(use_row_major_k_prefetch):

                    def _load_bundle(ks):
                        b_frags, b_scales, a_scales = _load_b_and_scales(
                            b_buf, b_bases, as_buf, as_bases, bs_buf, bs_bases, ks
                        )
                        a_frag = load_a_frag(a_buf, a_bases[0], ks)
                        return a_frag, b_frags, a_scales, b_scales

                    def _emit_bundle(bundle, emit_filler_now=False):
                        a_frag, b_frags, a_scales, b_scales = bundle
                        if const_expr(emit_filler_now and emit_filler is not None):
                            rocdl.sched_barrier(0)
                            emit_filler()
                        for wn in range_constexpr(wmma_n_rep):
                            _emit_wmma(
                                current_accs,
                                0,
                                wn,
                                a_frag,
                                b_frags[wn],
                                a_scales,
                                b_scales,
                            )

                    # Keep future K-subtile LDS reads outstanding while only draining
                    # the current bundle before its single row-major WMMA.
                    preload_depth = min(k_wmma_steps, _row_major_k_prefetch_depth + 1)
                    bundle_queue = [
                        _load_bundle(pre_ks)
                        for pre_ks in range_constexpr(preload_depth)
                    ]
                    next_ks = preload_depth
                    for ks in range_constexpr(k_wmma_steps):
                        is_last_ks = ks == k_wmma_steps - 1
                        cur_bundle = bundle_queue.pop(0)
                        rocdl.s_wait_dscnt(
                            len(bundle_queue) * _row_major_k_prefetch_bundle_ds
                        )

                        if const_expr(is_last_ks and late_compute_callback is not None):
                            rocdl.sched_barrier(0)
                            late_compute_callback()

                        _emit_bundle(cur_bundle, emit_filler_now=is_last_ks)

                        if const_expr(ks == 0 and mid_compute_callback is not None):
                            rocdl.sched_barrier(0)
                            mid_compute_callback()

                        if const_expr(next_ks < k_wmma_steps):
                            bundle_queue.append(_load_bundle(next_ks))
                            next_ks += 1

                    return current_accs

                prev_b, prev_bs, prev_as = _load_b_and_scales(
                    b_buf, b_bases, as_buf, as_bases, bs_buf, bs_bases, 0
                )
                for ks in range_constexpr(k_wmma_steps - 1):
                    _mid_cb = mid_compute_callback if ks == 0 else None
                    current_accs, (prev_b, prev_bs, prev_as) = _a_streaming_compute(
                        current_accs,
                        a_buf,
                        a_bases,
                        prev_b,
                        prev_bs,
                        prev_as,
                        ks,
                        next_bs_info=(
                            b_buf,
                            b_bases,
                            as_buf,
                            as_bases,
                            bs_buf,
                            bs_bases,
                            ks + 1,
                        ),
                        mid_compute_callback=_mid_cb,
                    )
                current_accs = _a_streaming_compute(
                    current_accs,
                    a_buf,
                    a_bases,
                    prev_b,
                    prev_bs,
                    prev_as,
                    k_wmma_steps - 1,
                    emit_filler=emit_filler,
                )
            return current_accs

        def compute_tile_fp8_quadrant(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
        ):
            current_accs = list(accs_in)
            _set_vgpr_a_scales(lds_as=lds_as)
            _set_blockscale_b_scales(lds_bs=lds_bs)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            # blockscale delivers scales via the VGPR box; bases unused.
            as_buf, as_bases = lds_as, None
            bs_buf, bs_bases = lds_bs, None
            _b_half_loads = _fp8_half_wn * _b_frag_loads_per_wn
            _b_left_bundle_loads = _b_half_loads + _fp8_b_scale_loads

            def _load_a_group(wm_base, wm_count, ks):
                return [
                    load_a_frag(a_buf, a_bases[wm_base + wm_local], ks)
                    for wm_local in range_constexpr(wm_count)
                ]

            def _load_b_half(wn_base, ks):
                return [
                    load_b_frag(b_buf, b_bases, wn_base + wn_local, ks)
                    for wn_local in range_constexpr(_fp8_half_wn)
                ]

            def _load_a_scales(ks):
                return _load_a_scale_operand(as_buf, as_bases, ks)

            def _load_b_scales(ks):
                return _load_b_scale_blockscale(ks)

            def _load_b_left_bundle(ks):
                return _load_b_half(0, ks), _load_b_scales(ks)

            def _emit_group_rows(
                wm_base,
                wn_base,
                a_frags,
                b_frags,
                a_scales,
                b_scales,
                row_start,
                row_count,
                emit_filler_now=False,
            ):
                if const_expr(emit_filler_now and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()
                for row_offset in range_constexpr(row_count):
                    wm_local = row_start + row_offset
                    global_wm = wm_base + wm_local
                    for wn_local in range_constexpr(_fp8_half_wn):
                        global_wn = wn_base + wn_local
                        _emit_wmma(
                            current_accs,
                            global_wm,
                            global_wn,
                            a_frags[wm_local],
                            b_frags[wn_local],
                            a_scales,
                            b_scales,
                        )

            def _emit_group(
                wm_base,
                wn_base,
                a_frags,
                b_frags,
                a_scales,
                b_scales,
                emit_filler_now=False,
            ):
                _emit_group_rows(
                    wm_base,
                    wn_base,
                    a_frags,
                    b_frags,
                    a_scales,
                    b_scales,
                    0,
                    _fp8_half_wm,
                    emit_filler_now=emit_filler_now,
                )

            def _emit_group_col(
                wm_base, wn_base, a_frags, b_frags, a_scales, b_scales, wn_local
            ):
                global_wn = wn_base + wn_local
                for wm_local in range_constexpr(_fp8_half_wm):
                    global_wm = wm_base + wm_local
                    _emit_wmma(
                        current_accs,
                        global_wm,
                        global_wn,
                        a_frags[wm_local],
                        b_frags[wn_local],
                        a_scales,
                        b_scales,
                    )

            b_left_frags, b_scales = _load_b_left_bundle(0)
            # Margin = a-top drain depth. blockscale delivers B-scale via VGPR
            # prefetch (0 ds loads), so the margin is just the b-scale count.
            _top_keep_margin = _fp8_b_scale_loads
            _first_top_row_keep = max(
                (_fp8_half_wm - 1) * DS_LOADS_PER_A_FRAG - _top_keep_margin, 0
            )
            _bottom_left_keep = max(_b_half_loads - DS_LOADS_PER_A_FRAG, 0)

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1
                a_scales = _load_a_scales(ks)

                a_top_frags = _load_a_group(0, _fp8_half_wm, ks)

                # Consume the first top-left row before issuing bottom-A.
                # The barriers only constrain LLVM scheduling; they are not
                # hardware synchronization points.
                rocdl.s_wait_dscnt(_first_top_row_keep)
                rocdl.sched_barrier(0)
                _emit_group_rows(
                    0, 0, a_top_frags, b_left_frags, a_scales, b_scales, 0, 1
                )
                rocdl.sched_barrier(0)

                a_bottom_frags = _load_a_group(_fp8_half_wm, _fp8_half_wm, ks)
                if const_expr(_fp8_half_wm > 1):
                    _emit_group_rows(
                        0,
                        0,
                        a_top_frags,
                        b_left_frags,
                        a_scales,
                        b_scales,
                        1,
                        _fp8_half_wm - 1,
                    )
                b_right_frags = _load_b_half(_fp8_half_wn, ks)

                # Drain bottom-A while keeping most right-half B in flight.
                rocdl.s_wait_dscnt(_bottom_left_keep)

                _emit_group(
                    _fp8_half_wm, 0, a_bottom_frags, b_left_frags, a_scales, b_scales
                )

                if const_expr(ks == 0 and mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                if const_expr(not is_last_ks):
                    next_left_frags, next_b_scales = _load_b_left_bundle(ks + 1)

                for wn_local in range_constexpr(_fp8_half_wn):
                    if const_expr(not is_last_ks):
                        _right_keep = (
                            _b_left_bundle_loads
                            + (_fp8_half_wn - wn_local - 1) * _b_frag_loads_per_wn
                        )
                    else:
                        _right_keep = (
                            _fp8_half_wn - wn_local - 1
                        ) * _b_frag_loads_per_wn
                    rocdl.s_wait_dscnt(_right_keep)
                    _emit_group_col(
                        0,
                        _fp8_half_wn,
                        a_top_frags,
                        b_right_frags,
                        a_scales,
                        b_scales,
                        wn_local,
                    )

                if const_expr(is_last_ks and late_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    late_compute_callback()

                if const_expr(is_last_ks and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()

                for wn_local in range_constexpr(_fp8_half_wn):
                    _emit_group_col(
                        _fp8_half_wm,
                        _fp8_half_wn,
                        a_bottom_frags,
                        b_right_frags,
                        a_scales,
                        b_scales,
                        wn_local,
                    )

                if const_expr(not is_last_ks):
                    b_left_frags = next_left_frags
                    b_scales = next_b_scales

            return current_accs

        def compute_tile_fp8_deep_pipeline(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
            a0_prefetch=None,
        ):
            current_accs = list(accs_in)
            _set_vgpr_a_scales(lds_as=lds_as)
            _set_blockscale_b_scales(lds_bs=lds_bs)
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b)
            # blockscale delivers scales via the VGPR box; bases unused.
            as_buf, as_bases = lds_as, None
            bs_buf, bs_bases = lds_bs, None

            def load_a_pair(wm_pair, ks):
                wm_base = wm_pair * _fp8_pair_wm
                return [
                    load_a_frag(a_buf, a_bases[wm_base + wm_local], ks)
                    for wm_local in range_constexpr(_fp8_pair_wm)
                ]

            def load_b_pair(wn_pair, ks):
                wn_base = wn_pair * _fp8_pair_wn
                return [
                    load_b_frag(b_buf, b_bases, wn_base + wn_local, ks)
                    for wn_local in range_constexpr(_fp8_pair_wn)
                ]

            def emit_panel_2x2(
                wm_pair,
                wn_pair,
                a_pair,
                b_pair,
                scale_pair,
                prefetch_after_first_row=None,
            ):
                a_scales, b_scales = scale_pair
                wm_base = wm_pair * _fp8_pair_wm
                wn_base = wn_pair * _fp8_pair_wn
                for wn_local in range_constexpr(_fp8_pair_wn):
                    _emit_wmma(
                        current_accs,
                        wm_base,
                        wn_base + wn_local,
                        a_pair[0],
                        b_pair[wn_local],
                        a_scales,
                        b_scales,
                    )
                if const_expr(prefetch_after_first_row is not None):
                    prefetch_after_first_row()
                for wn_local in range_constexpr(_fp8_pair_wn):
                    _emit_wmma(
                        current_accs,
                        wm_base + 1,
                        wn_base + wn_local,
                        a_pair[1],
                        b_pair[wn_local],
                        a_scales,
                        b_scales,
                    )

            def emit_panel_2x2_row(
                wm_pair, wn_pair, row_local, a_pair, b_pair, scale_pair
            ):
                a_scales, b_scales = scale_pair
                wm_base = wm_pair * _fp8_pair_wm
                wn_base = wn_pair * _fp8_pair_wn
                for wn_local in range_constexpr(_fp8_pair_wn):
                    _emit_wmma(
                        current_accs,
                        wm_base + row_local,
                        wn_base + wn_local,
                        a_pair[row_local],
                        b_pair[wn_local],
                        a_scales,
                        b_scales,
                    )

            _pair_loads = _fp8_pair_a_loads
            _two_pair_loads = _fp8_pair_a_loads + _fp8_pair_b_loads

            for ks in range_constexpr(k_wmma_steps):
                is_last_ks = ks == k_wmma_steps - 1
                a_scales, b_scales = _scales_for_emit(
                    as_buf, as_bases, bs_buf, bs_bases, ks
                )
                scale_pair = (a_scales, b_scales)

                b0 = load_b_pair(0, ks)
                if const_expr(
                    ks == 0
                    and a0_prefetch is not None
                    and len(a0_prefetch) == _fp8_pair_wm
                ):
                    a0 = list(a0_prefetch)
                elif const_expr(ks == 0 and a0_prefetch is not None):
                    a0 = [a0_prefetch[0], load_a_frag(a_buf, a_bases[1], ks)]
                else:
                    a0 = load_a_pair(0, ks)
                b1 = load_b_pair(1, ks)
                b2 = load_b_pair(2, ks)

                a1_box = [None]
                b3_box = [None]
                a2_box = [None]
                a3_box = [None]

                def _prefetch_a1():
                    a1_box[0] = load_a_pair(1, ks)

                first_wait_keep = _two_pair_loads + 3
                if const_expr(ks == 0 and a0_prefetch is not None):
                    first_wait_keep += DS_LOADS_PER_A_FRAG * len(a0_prefetch)
                rocdl.s_wait_dscnt(first_wait_keep)
                emit_panel_2x2(
                    0, 0, a0, b0, scale_pair, prefetch_after_first_row=_prefetch_a1
                )

                if const_expr(ks == 0 and mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()

                def _prefetch_b3():
                    b3_box[0] = load_b_pair(3, ks)

                def _prefetch_a3():
                    a3_box[0] = load_a_pair(3, ks)

                rocdl.s_wait_dscnt(_pair_loads + _fp8_pair_b_loads)
                emit_panel_2x2(
                    0, 1, a0, b1, scale_pair, prefetch_after_first_row=_prefetch_b3
                )

                rocdl.s_wait_dscnt(_fp8_pair_b_loads + 2)
                emit_panel_2x2(
                    1,
                    0,
                    a1_box[0],
                    b0,
                    scale_pair,
                    prefetch_after_first_row=_prefetch_a3,
                )

                def _prefetch_a2():
                    a2_box[0] = load_a_pair(2, ks)

                emit_panel_2x2(1, 1, a1_box[0], b1, scale_pair)

                emit_panel_2x2(
                    0, 2, a0, b2, scale_pair, prefetch_after_first_row=_prefetch_a2
                )
                emit_panel_2x2_row(1, 2, 0, a1_box[0], b2, scale_pair)
                emit_panel_2x2_row(1, 2, 1, a1_box[0], b2, scale_pair)
                rocdl.s_wait_dscnt(_pair_loads)
                emit_panel_2x2(0, 3, a0, b3_box[0], scale_pair)
                emit_panel_2x2(1, 3, a1_box[0], b3_box[0], scale_pair)

                emit_panel_2x2(2, 0, a2_box[0], b0, scale_pair)
                if const_expr(is_last_ks and late_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    late_compute_callback()
                emit_panel_2x2(2, 1, a2_box[0], b1, scale_pair)

                rocdl.s_wait_dscnt(0)
                emit_panel_2x2(3, 0, a3_box[0], b0, scale_pair)
                emit_panel_2x2(3, 1, a3_box[0], b1, scale_pair)

                if const_expr(is_last_ks and emit_filler is not None):
                    rocdl.sched_barrier(0)
                    emit_filler()

                emit_panel_2x2(2, 2, a2_box[0], b2, scale_pair)
                emit_panel_2x2(2, 3, a2_box[0], b3_box[0], scale_pair)
                emit_panel_2x2(3, 2, a3_box[0], b2, scale_pair)
                emit_panel_2x2(3, 3, a3_box[0], b3_box[0], scale_pair)

            return current_accs

        def hot_loop_scheduler():
            if const_expr(use_row_major_k_prefetch):
                _queue_depth = min(k_wmma_steps, _row_major_k_prefetch_depth + 1)
                for _ks in range_constexpr(k_wmma_steps):
                    if const_expr(_ks == 0):
                        rocdl.sched_dsrd(_row_major_k_prefetch_bundle_ds * _queue_depth)
                    elif const_expr(_ks + _queue_depth <= k_wmma_steps):
                        rocdl.sched_dsrd(_row_major_k_prefetch_bundle_ds)
                    rocdl.sched_mfma(wmma_n_rep)
                rocdl.sched_barrier(0)
                return

            _half_wm = wmma_m_rep // 2
            _half_wmma = _half_wm * wmma_n_rep
            _b_loads_per_frag = 4
            _scale_dsrd = _scale_ds_loads
            _a_half_dsrd = _half_wm * DS_LOADS_PER_A_FRAG

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(
                        wmma_n_rep * _b_loads_per_frag + _scale_dsrd + _a_half_dsrd
                    )
                else:
                    rocdl.sched_dsrd(_a_half_dsrd)
                rocdl.sched_mfma(_half_wmma)
                rocdl.sched_dsrd(_a_half_dsrd)
                rocdl.sched_mfma(_half_wmma)
                if const_expr(_ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(wmma_n_rep * _b_loads_per_frag + _scale_dsrd)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_fp8_quadrant():
            _a_scale_loads = 0
            _a_top_loads = _fp8_half_wm * DS_LOADS_PER_A_FRAG
            _a_bottom_loads = _a_top_loads
            _b_half_loads = _fp8_half_wn * _b_frag_loads_per_wn
            _b_left_bundle_loads = _b_half_loads + _fp8_b_scale_loads
            _group_wmma = _fp8_group_size
            _first_row_wmma = _fp8_half_wn
            _remaining_top_left_wmma = (_fp8_half_wm - 1) * _fp8_half_wn

            for _ks in range_constexpr(k_wmma_steps):
                if const_expr(_ks == 0):
                    rocdl.sched_dsrd(
                        _b_left_bundle_loads + _a_scale_loads + _a_top_loads
                    )
                else:
                    rocdl.sched_dsrd(_a_scale_loads + _a_top_loads)
                rocdl.sched_mfma(_first_row_wmma)
                rocdl.sched_dsrd(_a_bottom_loads)
                if const_expr(_remaining_top_left_wmma > 0):
                    rocdl.sched_mfma(_remaining_top_left_wmma)
                rocdl.sched_dsrd(_b_half_loads)
                rocdl.sched_mfma(_group_wmma)
                if const_expr(_ks < k_wmma_steps - 1):
                    rocdl.sched_dsrd(_b_left_bundle_loads)
                for _wn_local in range_constexpr(_fp8_half_wn):
                    rocdl.sched_mfma(_fp8_half_wm)
                for _wn_local in range_constexpr(_fp8_half_wn):
                    rocdl.sched_mfma(_fp8_half_wm)
            rocdl.sched_barrier(0)

        def hot_loop_scheduler_fp8_deep_pipeline():
            def _sched_panel_2x2(prefetch_loads=0):
                if const_expr(prefetch_loads > 0):
                    rocdl.sched_mfma(_fp8_pair_wn)
                    rocdl.sched_dsrd(prefetch_loads)
                    rocdl.sched_mfma(_fp8_pair_wn)
                else:
                    rocdl.sched_mfma(_fp8_pair_wm * _fp8_pair_wn)

            def _sched_panel_row():
                rocdl.sched_mfma(_fp8_pair_wn)

            _initial_loads = (
                _fp8_scale_loads + _fp8_pair_b_loads * 3 + _fp8_pair_a_loads
            )

            for _ks in range_constexpr(k_wmma_steps):
                _ks_initial_loads = _initial_loads
                if const_expr(_ks == 0):
                    _ks_initial_loads -= _fp8_pair_a_loads
                rocdl.sched_dsrd(_ks_initial_loads)
                _sched_panel_2x2(_fp8_pair_a_loads)
                _sched_panel_2x2(_fp8_pair_b_loads)
                _sched_panel_2x2(_fp8_pair_a_loads)
                _sched_panel_2x2()
                _sched_panel_2x2(_fp8_pair_a_loads)
                _sched_panel_row()
                _sched_panel_row()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
                _sched_panel_2x2()
            rocdl.sched_barrier(0)

        def compute_tile_scheduled(
            accs_in,
            lds_a,
            lds_b,
            lds_as,
            lds_bs,
            emit_filler=None,
            mid_compute_callback=None,
            late_compute_callback=None,
            a0_prefetch=None,
        ):
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT):
                return compute_tile_fp8_quadrant(
                    accs_in,
                    lds_a,
                    lds_b,
                    lds_as,
                    lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                    late_compute_callback=late_compute_callback,
                )
            if const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE):
                return compute_tile_fp8_deep_pipeline(
                    accs_in,
                    lds_a,
                    lds_b,
                    lds_as,
                    lds_bs,
                    emit_filler=emit_filler,
                    mid_compute_callback=mid_compute_callback,
                    late_compute_callback=late_compute_callback,
                    a0_prefetch=a0_prefetch,
                )
            return compute_tile(
                accs_in,
                lds_a,
                lds_b,
                lds_as,
                lds_bs,
                emit_filler=emit_filler,
                mid_compute_callback=mid_compute_callback,
                late_compute_callback=late_compute_callback,
            )

        def hot_loop_scheduler_scheduled():
            if const_expr(
                compute_schedule_kind == COMPUTE_SCHEDULE_FP8_DEEP_PIPELINE
            ):
                hot_loop_scheduler_fp8_deep_pipeline()
            elif const_expr(compute_schedule_kind == COMPUTE_SCHEDULE_FP8_QUADRANT):
                hot_loop_scheduler_fp8_quadrant()
            else:
                hot_loop_scheduler()

        def prefetch_fp8_deep_a0_frags(lds_a):
            a_buf, a_bases = _precompute_a_lane_bases(lds_a)
            return [
                load_a_frag(a_buf, a_bases[wm_local], 0)
                for wm_local in range_constexpr(_fp8_pair_wm)
            ]

        def maybe_prefetch_fp8_deep_a0(lds_a):
            # Call only after the TDM fence for this stage; pre-fence LDS reads can race multicast delivery.
            if const_expr(use_fp8_deep_pipeline_schedule):
                return prefetch_fp8_deep_a0_frags(lds_a)
            return None

        # ── Epilogue (unified via _sub_tiles) ──
        def _get_acc_sub8(accs, acc_idx, vec_base):
            """Extract 8-element sub-vector from accumulator."""
            if const_expr(ACC_VEC_SIZE == 8):
                return accs[acc_idx]
            indices = [vec_base + i for i in range_constexpr(8)]
            acc = fx.Vector(accs[acc_idx])
            return acc.shuffle(acc, indices)

        def epilogue_prepare_addrs():
            addrs = []
            _bf16_out = out_dtype in ("bf16", "f16")
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                row = blk_m + warp_m_base + arith.index(m_off) + lane16
                col_base = (
                    blk_n
                    + warp_n_base
                    + arith.index(wn * WMMA_N)
                    + lane_kgrp * arith.index(8)
                )
                if const_expr(_bf16_out):
                    c_off_bytes = (
                        row * n_stride + col_base + c_batch_off
                    ) * arith.index(elem_bytes_d)
                    addrs.append(c_off_bytes)
                else:
                    for half in range_constexpr(2):
                        col = col_base + arith.index(half * 4)
                        c_off = row * n_stride + col + c_batch_off
                        addrs.append(c_off)
            return addrs

        _bf16_out = out_dtype in ("bf16", "f16")
        _out_elem_local = (
            T.bf16 if out_dtype == "bf16" else (T.f16 if out_dtype == "f16" else None)
        )

        def epilogue_stores(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                n_slots = 1 if _bf16_out else 2
                # This path only runs for the partial last M-tile. Rows >= M are
                # overhang and must be skipped per-lane: c_rsrc's num_records spans
                # ALL batches (B*M*N), so for the contiguous BMN [B,M,N] layout an
                # overhang row of batch b would write into batch b+1 instead of
                # being clipped (only the final batch's overhang exceeds
                # num_records). Guard on row < M, matching epilogue_atomic_adds.
                row = blk_m + warp_m_base + arith.index(m_off) + lane16
                if_op = scf.IfOp(row < m_idx, [], has_else=False)
                with ir.InsertionPoint(if_op.then_block):
                    if const_expr(_bf16_out):
                        store_acc_vec8_to_buffer(
                            sub8,
                            c_rsrc,
                            addrs[addr_idx],
                            out_elem=_out_elem_local,
                            offset_is_bytes=True,
                        )
                    else:
                        store_acc_vec8_to_buffer(
                            sub8, c_rsrc, addrs[addr_idx : addr_idx + 2]
                        )
                    scf.YieldOp([])
                addr_idx += n_slots

        def epilogue_lds_stores(final_accs, d_buf, d_base):
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                imm = m_off * _lds_d_stride_elems + wn * _n_col_d_elems
                store_acc_vec8_to_lds(
                    d_buf, d_base, imm, sub8, out_elem=_out_elem_local
                )

        def _atomic_fadd_global(val, byte_off):
            # Device-scoped, relaxed atomic add into C at c_global_base_i64 + byte_off.
            addr_i64 = llvm.AddOp(
                c_global_base_i64,
                arith.index_cast(T.i64, byte_off),
                llvm.IntegerOverflowFlags(0),
            ).result
            ptr = llvm.IntToPtrOp(c_global_ptr_type, addr_i64).result
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.fadd,
                ptr,
                val.ir_value(),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )

        def _atomic_add_acc_vec8_to_buffer(acc_vec8, addr):
            if const_expr(_bf16_out):
                h_vec = fx.Vector(arith.trunc_f(T.vec(8, _out_elem_local), acc_vec8))
                for pair in range_constexpr(4):
                    pair_vec = fx.Vector.from_elements(
                        [h_vec[pair * 2], h_vec[pair * 2 + 1]]
                    )
                    byte_off = addr + arith.index(pair * 4)
                    _atomic_fadd_global(pair_vec, byte_off)
                return 1

            acc_vec = fx.Vector(acc_vec8)
            for half in range_constexpr(2):
                base_addr = addr[half] if isinstance(addr, (list, tuple)) else addr
                for vi in range_constexpr(4):
                    val = acc_vec[half * 4 + vi]
                    byte_off = (base_addr + arith.index(vi)) * arith.index(4)
                    _atomic_fadd_global(val, byte_off)
            return 2

        def epilogue_atomic_adds(final_accs, addrs):
            addr_idx = 0
            for acc_idx, vec_base, m_off, wn in _sub_tiles:
                sub8 = _get_acc_sub8(final_accs, acc_idx, vec_base)
                n_slots = 1 if _bf16_out else 2
                addr_arg = (
                    addrs[addr_idx] if _bf16_out else addrs[addr_idx : addr_idx + 2]
                )
                # Atomics use a raw global ptr (no num_records clip), so predicate
                # per-lane to skip rows >= M.
                row = blk_m + warp_m_base + arith.index(m_off) + lane16
                if_op = scf.IfOp(row < m_idx, [], has_else=False)
                with ir.InsertionPoint(if_op.then_block):
                    _atomic_add_acc_vec8_to_buffer(sub8, addr_arg)
                    scf.YieldOp([])
                addr_idx += n_slots

        _effective_l2_pf = l2_prefetch_distance
        if const_expr(use_cluster and l2_prefetch_distance > 0):
            _effective_l2_pf = max(1, l2_prefetch_distance - 1)

        def _l2_prefetch(k_base):
            if const_expr(_effective_l2_pf <= 0):
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            # Match the TDM descriptors: batch rides the unit-stride inner dim.
            pf_k_packed_a = pf_k + a_batch_off
            pf_k_packed_b = pf_k
            tdm_ops.l2_prefetch_tile(
                arg_a,
                (blk_m, pf_k_packed_a),
                (tile_m, packed_tile_k_a),
                #runtime value
                (lda_packed, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )
            tdm_ops.l2_prefetch_tile(
                arg_b,
                (
                    blk_n // arith.index(16),
                    pf_k_packed_b * arith.index(16) + b_batch_off,
                ),
                (tile_n // 16, packed_tile_k_b * 16),
                (K_packed_b * 16, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )

        # ====== Multi-stage pipeline ======
        acc_zero = arith.constant_vector(0.0, T.vec(ACC_VEC_SIZE, T.f32))
        accs = [acc_zero] * n_accs

        lds_a_data_f16 = lds_a_data_bytes // 2
        lds_b_data_f16 = lds_b_data_bytes // 2
        lds_a_scale_f16 = lds_a_scale_bytes // 2
        lds_b_scale_f16 = lds_b_scale_bytes // 2

        arena_base_ptr = arena_alloc.get_base()

        stages_a = [
            SmemPtr(
                arena_base_ptr,
                stage_a_data_off[i],
                elem_ty_lds,
                shape=(lds_a_data_f16,),
            )
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(
                arena_base_ptr,
                stage_b_data_off[i],
                elem_ty_lds,
                shape=(lds_b_data_f16,),
            )
            for i in range_constexpr(num_buffers)
        ]
        stages_as = [
            SmemPtr(
                arena_base_ptr,
                stage_a_scale_off[i],
                elem_ty_lds,
                shape=(lds_a_scale_f16,),
            )
            for i in range_constexpr(num_buffers)
        ]
        stages_bs = [
            SmemPtr(
                arena_base_ptr,
                stage_b_scale_off[i],
                elem_ty_lds,
                shape=(lds_b_scale_f16,),
            )
            for i in range_constexpr(num_buffers)
        ]

        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_as_mem = [stages_as[i].get() for i in range_constexpr(num_buffers)]
        stages_bs_mem = [stages_bs[i].get() for i in range_constexpr(num_buffers)]

        stages_a_idx = [
            extract_lds_base_idx(stages_a[i]) for i in range_constexpr(num_buffers)
        ]
        stages_b_idx = [
            extract_lds_base_idx(stages_b[i]) for i in range_constexpr(num_buffers)
        ]
        stages_as_idx = [
            extract_lds_base_idx(stages_as[i]) for i in range_constexpr(num_buffers)
        ]
        stages_bs_idx = [
            extract_lds_base_idx(stages_bs[i]) for i in range_constexpr(num_buffers)
        ]

        if const_expr(tdm_store_enabled):
            d_lds_base_ptr = arena_base_ptr
            d_lds_f16_count = total_d_bytes // 2
            d_smem = SmemPtr(
                d_lds_base_ptr, d_output_off, elem_ty_lds, shape=(d_lds_f16_count,)
            )
            d_lds_buffer = get_lds_memref(d_smem)
            warp_lds_off = (
                wave_m_idx * arith.index(n_warp) + wave_n_idx
            ) * arith.index(_warp_d_elems)
            d_lane_base = (
                warp_lds_off
                + lane16 * arith.index(_lds_d_stride_elems)
                + lane_kgrp * arith.index(4 * elem_bytes_d)
            )
            wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
            # Match the TDM-store descriptor offsets to the compute wave mapping.
            if const_expr(use_fp8_deep_pipeline_schedule):
                wave_m_sgpr = wave_id_idx % arith.index(m_warp)
                wave_n_sgpr = wave_id_idx // arith.index(m_warp)
            else:
                wave_m_sgpr = wave_id_idx // arith.index(n_warp)
                wave_n_sgpr = wave_id_idx % arith.index(n_warp)
            d_warp_linear_sgpr = wave_m_sgpr * arith.index(n_warp) + wave_n_sgpr
            d_warp_off_sgpr = d_warp_linear_sgpr * arith.index(
                warp_d_bytes
            ) + arith.index(d_output_off)
            warp_m_off_sgpr = wave_m_sgpr * arith.index(warp_tile_m)
            warp_n_off_sgpr = wave_n_sgpr * arith.index(warp_tile_n)
            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_c,
                lds_memref=d_lds_base_ptr,
                global_offset=(
                    blk_m + warp_m_off_sgpr,
                    blk_n + warp_n_off_sgpr + c_batch_off,
                ),
                tensor_shape=(warp_tile_m, warp_tile_n),
                strides=(n_stride, 1),
                tile_shape=(warp_tile_m, warp_tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=warp_tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
                oob_outer_bound=i32_m,
            )

        # TDM descriptor lane layout: dgroup0 = [predicate, lds_addr, addr_lo, addr_hi].
        def _dg0_lane(desc, lane):
            return fx.Vector(desc.dgroup0)[lane]

        def _pack_dg0(pred, lds_addr, addr_lo, addr_hi):
            return fx.Vector.from_elements([pred, lds_addr, addr_lo, addr_hi], fx.Int32)

        # Precompute LDS addresses for TDM descriptor switching
        stages_a_lds_addr = []
        stages_b_lds_addr = []
        stages_as_lds_addr = []
        stages_bs_lds_addr = []
        for i in range_constexpr(num_buffers):
            stages_a_lds_addr.append(
                _dg0_lane(make_desc_a(stages_a_mem[i], arith.index(0)), 1)
            )
            stages_b_lds_addr.append(
                _dg0_lane(make_desc_b(stages_b_mem[i], arith.index(0)), 1)
            )
            stages_as_lds_addr.append(
                _dg0_lane(make_desc_as(stages_as_mem[i], arith.index(0)), 1)
            )
            stages_bs_lds_addr.append(
                _dg0_lane(make_desc_bs(stages_bs_mem[i], arith.index(0)), 1)
            )

        desc_a_init = make_desc_a(stages_a_mem[0], split_k_base)
        desc_b_init = make_desc_b(stages_b_mem[0], split_k_base)
        desc_as_init = make_desc_as(stages_as_mem[0], split_k_base)
        desc_bs_init = make_desc_bs(stages_bs_mem[0], split_k_base)

        adv_a_i32 = fx.Int32(tile_k)
        adv_b_i32 = fx.Int32(packed_tile_k_b * 16)
        if const_expr(ascale_col_major):
            adv_as_i32 = arith.index_cast(
                T.i32, arith.index(k_wmma_steps) * stride_ascale_k
            )
        else:
            adv_as_i32 = fx.Int32(bsc_a_row_stride_bytes)
        adv_bs_i32 = fx.Int32(bsc_b_row_stride_bytes)

        _active_wave_limit = min(num_warps, 4)
        active_pred_const = (
            fx.Int32(1)
            if _active_wave_limit >= num_warps
            else arith.select(
                tdm_wave_id < fx.Int32(_active_wave_limit), fx.Int32(1), fx.Int32(0)
            )
        )

        def _select4(values):
            return _select_wave_tdm_value(values[0], values[1], values[2], values[3])

        def _desc_lanes(descs, lane):
            return [_dg0_lane(desc, lane) for desc in descs]

        def _select_active_tdm(stage_lds_addrs, descs, advs):
            active_stages = [
                _select_wave_tdm_value(
                    stage_lds_addrs[0][i],
                    stage_lds_addrs[1][i],
                    stage_lds_addrs[2][i],
                    stage_lds_addrs[3][i],
                )
                for i in range_constexpr(num_buffers)
            ]
            return (
                active_stages,
                _select4(_desc_lanes(descs, 2)),
                _select4(_desc_lanes(descs, 3)),
                _select4([desc.dgroup1 for desc in descs]),
                _select4(advs),
            )

        _tdm_stage_sel = (
            stages_a_lds_addr,
            stages_b_lds_addr,
            stages_as_lds_addr,
            stages_bs_lds_addr,
        )
        _tdm_desc_sel = (desc_a_init, desc_b_init, desc_as_init, desc_bs_init)
        _tdm_adv_sel = (adv_a_i32, adv_b_i32, adv_as_i32, adv_bs_i32)
        (
            active_stage_lds_addr,
            active_addr_lo,
            active_addr_hi,
            active_dgroup1,
            active_adv_i32,
        ) = _select_active_tdm(_tdm_stage_sel, _tdm_desc_sel, _tdm_adv_sel)
        if const_expr(secondary_scale_tdm):
            if const_expr(two_wave_scale):
                sec_pred_const = arith.select(
                    tdm_wave_id < fx.Int32(2), fx.Int32(1), fx.Int32(0)
                )
                sec_stage_lds_addr = [
                    arith.select(
                        tdm_wave_is_a, stages_bs_lds_addr[i], stages_as_lds_addr[i]
                    )
                    for i in range_constexpr(num_buffers)
                ]
                sec_addr_hi = arith.select(
                    tdm_wave_is_a,
                    _dg0_lane(desc_bs_init, 3),
                    _dg0_lane(desc_as_init, 3),
                )
                sec_dgroup1 = arith.select(
                    tdm_wave_is_a, desc_bs_init.dgroup1, desc_as_init.dgroup1
                )
                sec_adv_i32 = arith.select(tdm_wave_is_a, adv_bs_i32, adv_as_i32)
                sec_addr_lo_init = arith.select(
                    tdm_wave_is_a,
                    _dg0_lane(desc_bs_init, 2),
                    _dg0_lane(desc_as_init, 2),
                )
            else:
                # 3-wave compatibility: wave2 carries A-scale, wave0 carries B-scale.
                sec_pred_const = arith.select(tdm_wave_is_a, fx.Int32(1), fx.Int32(0))
                sec_stage_lds_addr = stages_bs_lds_addr
                sec_addr_hi = _dg0_lane(desc_bs_init, 3)
                sec_dgroup1 = desc_bs_init.dgroup1
                sec_adv_i32 = adv_bs_i32
                sec_addr_lo_init = _dg0_lane(desc_bs_init, 2)

        def _pipeline_fence(outstanding=0):
            pipeline_fence(outstanding=outstanding, use_cluster=use_cluster)

        def _pipeline_fence_signal(outstanding=0):
            pipeline_fence_signal(outstanding=outstanding, use_cluster=use_cluster)

        def _issue_active_tdm(load_stage, addr_box, k_prefetch=None, sec_box=None):
            dg0 = _pack_dg0(
                active_pred_const,
                active_stage_lds_addr[load_stage],
                addr_box[0],
                active_addr_hi,
            )
            tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0, active_dgroup1))
            addr_box[0] = addr_box[0] + active_adv_i32
            if const_expr(secondary_scale_tdm):
                dg0s = _pack_dg0(
                    sec_pred_const,
                    sec_stage_lds_addr[load_stage],
                    sec_box[0],
                    sec_addr_hi,
                )
                tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(dg0s, sec_dgroup1))
                sec_box[0] = sec_box[0] + sec_adv_i32
            if k_prefetch is not None:
                _l2_prefetch(k_prefetch)

        # Prologue
        if const_expr(secondary_scale_tdm):
            active_sec_lo = sec_addr_lo_init
        for i in range_constexpr(pre_loaded):
            addr_box = [active_addr_lo]
            if const_expr(secondary_scale_tdm):
                sec_box = [active_sec_lo]
                _issue_active_tdm(i, addr_box, sec_box=sec_box)
                active_sec_lo = sec_box[0]
            else:
                _issue_active_tdm(i, addr_box)
            active_addr_lo = addr_box[0]

        _pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2))

        # Main loop — acc_mixed style: fence at top, TDM_load mid-compute.
        # This overlaps TDM DMA with the remaining WMMA instructions,
        _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)

        if const_expr(loop_iters > 0 and use_tdm_late_signal_overlap):
            _pipeline_fence_signal(outstanding=_fence_outstanding)

        if const_expr(loop_iters > 0):
            init_args = list(accs) + [active_addr_lo]
            if const_expr(secondary_scale_tdm):
                init_args = init_args + [active_sec_lo]

            for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                accs_in = list(state[:n_accs])
                cur_addr_lo = state[n_accs]
                _state_off = n_accs + 1
                if const_expr(secondary_scale_tdm):
                    cur_sec_lo = state[_state_off]
                    _state_off = _state_off + 1

                for buf_idx in range_constexpr(num_buffers):
                    load_stage = (buf_idx + num_buffers - 1) % num_buffers
                    addr_box = [cur_addr_lo]
                    sec_box = [cur_sec_lo] if secondary_scale_tdm else None

                    def _mid_tdm_ws(
                        _ls=load_stage,
                        _ab=addr_box,
                        _sb=sec_box,
                        _k_off=(
                            split_k_base
                            + loop_iter * arith.index(num_buffers * tile_k)
                            + arith.index(buf_idx * tile_k)
                        ),
                    ):
                        _issue_active_tdm(_ls, _ab, k_prefetch=_k_off, sec_box=_sb)

                    if const_expr(not use_tdm_late_signal_overlap):
                        _pipeline_fence_signal(outstanding=_fence_outstanding)
                    pipeline_fence_wait(use_cluster=use_cluster)

                    _late_tdm_ws_fence_signal = None
                    if const_expr(use_tdm_late_signal_overlap):

                        def _late_tdm_ws_split_signal():
                            _pipeline_fence_signal(outstanding=_fence_outstanding)

                        _late_tdm_ws_fence_signal = _late_tdm_ws_split_signal

                    a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[buf_idx])
                    rocdl.sched_barrier(0)

                    accs_in = compute_tile_scheduled(
                        accs_in,
                        stages_a_idx[buf_idx],
                        stages_b_idx[buf_idx],
                        stages_as_idx[buf_idx],
                        stages_bs_idx[buf_idx],
                        mid_compute_callback=_mid_tdm_ws,
                        late_compute_callback=_late_tdm_ws_fence_signal,
                        a0_prefetch=a0_prefetch,
                    )
                    cur_addr_lo = addr_box[0]
                    if const_expr(secondary_scale_tdm):
                        cur_sec_lo = sec_box[0]
                    hot_loop_scheduler_scheduled()

                _sec_yield = [cur_sec_lo] if secondary_scale_tdm else []
                results = yield list(accs_in) + [cur_addr_lo] + _sec_yield

            accs = list(results[:n_accs])
            active_addr_lo = results[n_accs]
            _result_off = n_accs + 1
            if const_expr(secondary_scale_tdm):
                active_sec_lo = results[n_accs + 1]
                _result_off = _result_off + 1
        # Tail — same acc_mixed pattern: fence at top, TDM mid-compute.
        if const_expr(loop_iters > 0 and use_tdm_late_signal_overlap):
            pipeline_fence_wait(use_cluster=use_cluster)
        if const_expr(loop_iters > 0):
            _pipeline_fence(outstanding=0)
        elif const_expr(use_cluster):
            cluster.cluster_barrier()
        epi_addrs_box = [None]
        _tail_had_load = False

        for _load_stage, _compute_stage, _outstanding in tail_plan:
            if const_expr(_outstanding == -1):
                if const_expr(_tail_had_load):
                    _pipeline_fence(outstanding=0)
                if const_expr(tdm_store_enabled):
                    a0_prefetch = maybe_prefetch_fp8_deep_a0(
                        stages_a_idx[_compute_stage]
                    )
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        a0_prefetch=a0_prefetch,
                    )
                else:

                    def _emit_epi_addrs():
                        epi_addrs_box[0] = epilogue_prepare_addrs()

                    a0_prefetch = maybe_prefetch_fp8_deep_a0(
                        stages_a_idx[_compute_stage]
                    )
                    accs = compute_tile_scheduled(
                        accs,
                        stages_a_idx[_compute_stage],
                        stages_b_idx[_compute_stage],
                        stages_as_idx[_compute_stage],
                        stages_bs_idx[_compute_stage],
                        emit_filler=_emit_epi_addrs,
                        a0_prefetch=a0_prefetch,
                    )
            else:
                _pipeline_fence_signal(outstanding=_outstanding)
                pipeline_fence_wait(use_cluster=use_cluster)

                _tail_mid_cb = None
                if const_expr(_load_stage is not None):
                    _tail_had_load = True
                    _tail_addr_box = [active_addr_lo]
                    _tail_sec_box = [active_sec_lo] if secondary_scale_tdm else None

                    def _tail_mid_ws(
                        _ls=_load_stage, _ab=_tail_addr_box, _sb=_tail_sec_box
                    ):
                        _issue_active_tdm(_ls, _ab, sec_box=_sb)

                    _tail_mid_cb = _tail_mid_ws

                a0_prefetch = maybe_prefetch_fp8_deep_a0(stages_a_idx[_compute_stage])
                rocdl.sched_barrier(0)
                accs = compute_tile_scheduled(
                    accs,
                    stages_a_idx[_compute_stage],
                    stages_b_idx[_compute_stage],
                    stages_as_idx[_compute_stage],
                    stages_bs_idx[_compute_stage],
                    mid_compute_callback=_tail_mid_cb,
                    a0_prefetch=a0_prefetch,
                )

                if const_expr(_load_stage is not None):
                    active_addr_lo = _tail_addr_box[0]
                    if const_expr(secondary_scale_tdm):
                        active_sec_lo = _tail_sec_box[0]

                hot_loop_scheduler_scheduled()

        def _emit_tdm_store():
            if const_expr(d_need_epilogue_fence):
                _pipeline_fence(outstanding=0)
            rocdl.sched_barrier(0)
            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)

        def _emit_buffer_store():
            rocdl.sched_barrier(0)
            if const_expr(epi_addrs_box[0] is None):
                epi_addrs_box[0] = epilogue_prepare_addrs()
            if const_expr(split_k > 1):
                epilogue_atomic_adds(accs, epi_addrs_box[0])
            else:
                epilogue_stores(accs, epi_addrs_box[0])

        if const_expr(tdm_store_enabled):
            full_tile = (blk_m + arith.index(tile_m)) <= m_idx
            if_op = scf.IfOp(full_tile, [], has_else=True)
            with ir.InsertionPoint(if_op.then_block):
                _emit_tdm_store()
                scf.YieldOp([])
            with ir.InsertionPoint(if_op.else_block):
                _emit_buffer_store()
                scf.YieldOp([])
        else:
            _emit_buffer_store()

    cache_tag = (
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        num_buffers,
        compute_schedule_kind,
        effective_waves_per_eu,
        l2_prefetch_distance,
        cluster_m,
        cluster_n,
        tdm_store_enabled,
        out_dtype,
        inst_prefetch,
        split_k,
        expert_sched_mode,
        atomic_barrier_enable,
        scale_block_k,
        scale_block_n,
        ascale_layout,
        _row_major_k_prefetch_depth,
    )

    def _emit_launch(
        arg_c,
        arg_a,
        arg_b,
        arg_a_scale,
        arg_b_scale,
        i32_batch,
        i32_m,
        i32_n,
        i32_lda,
        i32_ldc,
        i32_stride_ascale_m,
        i32_stride_ascale_k,
        i32_stride_a_batch,
        i32_stride_b_batch,
        i32_stride_c_batch,
        i32_stride_ascale_batch,
        i32_stride_bscale_batch,
        stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = N // tile_n
        gz = i32_batch * split_k

        if const_expr(use_cluster):
            gx = ((gx + (cluster_m - 1)) // cluster_m) * cluster_m

        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        kernel_blockscale_w8a8_bmm(
            arg_c,
            arg_a,
            arg_b,
            arg_a_scale,
            arg_b_scale,
            i32_batch,
            i32_m,
            i32_n,
            i32_lda,
            i32_ldc,
            i32_stride_ascale_m,
            i32_stride_ascale_k,
            i32_stride_a_batch,
            i32_stride_b_batch,
            i32_stride_c_batch,
            i32_stride_ascale_batch,
            i32_stride_bscale_batch,
            value_attrs={
                "rocdl.waves_per_eu": effective_waves_per_eu,
                "rocdl.cluster_dims": (
                    f"{cluster_m},{cluster_n},1" if const_expr(use_cluster) else None
                ),
            },
        ).launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
            cluster=cluster_arg,
        )

    @flyc.jit
    def launch_blockscale_w8a8_bmm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_batch: fx.Int32,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
        i32_stride_ascale_m: fx.Int32,
        i32_stride_ascale_k: fx.Int32,
        i32_stride_a_batch: fx.Int32,
        i32_stride_b_batch: fx.Int32,
        i32_stride_c_batch: fx.Int32,
        i32_stride_ascale_batch: fx.Int32,
        i32_stride_bscale_batch: fx.Int32,
        stream: fx.Stream,
    ):
        _emit_launch(
            arg_c,
            arg_a,
            arg_b,
            arg_a_scale,
            arg_b_scale,
            i32_batch,
            i32_m,
            i32_n,
            i32_lda,
            i32_ldc,
            i32_stride_ascale_m,
            i32_stride_ascale_k,
            i32_stride_a_batch,
            i32_stride_b_batch,
            i32_stride_c_batch,
            i32_stride_ascale_batch,
            i32_stride_bscale_batch,
            stream,
        )

    if effective_expert_sched_mode:
        launch_blockscale_w8a8_bmm.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
        }

    return launch_blockscale_w8a8_bmm


__all__ = [
    "compile_blockscale_w8a8_bmm",
]