
import functools
import inspect
import os

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, tdm_ops, vector, math
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr, check_smem_capacity
from flydsl.expr import idx2crd

from kernels.gemm_common_gfx1250 import (
    extract_lds_base_idx,
    get_lds_memref,
    issue_tdm_loads,
    lds_load_b128_raw,
    lds_store_b128,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    store_acc_vec8_to_buffer,
    store_acc_vec8_to_lds,
    workgroup_barrier,
)

from kernels.pipeline_utils import make_tail_plan, tdm_epilogue_fence_threshold_bytes

_TDM_HAS_EARLY_TIMEOUT = "early_timeout" in inspect.signature(tdm_ops.make_tensor_descriptor_2d).parameters

def _make_tdm_desc(*, early_timeout=False, **kwargs):
    if _TDM_HAS_EARLY_TIMEOUT:
        kwargs["early_timeout"] = early_timeout
    return tdm_ops.make_tensor_descriptor_2d(**kwargs)


WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
ACC_VEC_SIZE = 8
DS_LOADS_PER_A_FRAG = 4

LDS_PAD_A_BYTES = 16
LDS_PAD_B_BYTES = 16
LDS_PAD_D_BYTES = 16

ELEM_BYTES_A = 1  # fp8
ELEM_BYTES_B = 1  # fp8
ELEM_BYTES_SCALE = 1  # fp8

LDS_SEGMENT_BYTES = 64 * 1024
LDS_GFX1250_MAX_BYTES = 5 * LDS_SEGMENT_BYTES

@functools.lru_cache(maxsize=256)
def compile_bmm_w8a8_preshuffle_gfx1250(
    *,
    B: int = 16,
    M: int = 0,
    N: int = 1024,
    K: int = 4096,
    group_k: int = 128,
    group_n: int = 128,
    tile_m: int = 128,
    tile_n: int = 256,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    out_dtype: str = "bf16",
    num_buffers: int = 4,
    waves_per_eu: int = None,
    l2_prefetch_distance: int = 2,
    use_tdm_store: bool = True,
    expert_sched_mode: bool = True,
    inst_prefetch: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    wave_specialized_tdm: bool = False,
    atomic_barrier_enable: bool = False,
):
    
    _ = M
    wmma_op = rocdl.wmma_scale_f32_16x16x128_f8f6f4

    if out_dtype not in ("f32", "f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f32', 'bf16', or 'f16', got {out_dtype!r}")
    if tile_k % group_k != 0:
        raise ValueError(f"tile_k ({tile_k}) must be divisible by scale_block_k ({group_k})")
    if tile_n % group_n != 0:
        raise ValueError(f"tile_k ({tile_n}) must be divisible by scale_block_n ({group_n})")
    if K % tile_k != 0:
        raise ValueError(f"K ({K}) must be divisible by tile_k ({tile_k})")
    if N % tile_n != 0:
        raise ValueError(f"K ({N}) must be divisible by tile_n ({tile_n})")
    

    use_cluster = cluster_m > 1 or cluster_n > 1
    if const_expr(use_cluster):
        if cluster_m * cluster_n > 16:
            raise ValueError(f"cluster_m * cluster_n must be <= 16, got {cluster_m}*{cluster_n}")
        
    elem_bytes_d = 2 if out_dtype in ("bf16", "f16") else 4
    _effective_l2_pf = max(1, l2_prefetch_distance - 1) if use_cluster else l2_prefetch_distance
    effective_waves_per_eu = waves_per_eu

    num_warps = m_warp * n_warp
    block_threads = num_warps * WAVE_SIZE

    if block_threads > 1024:
        raise ValueError(f"block_threads must be <= 1024, got {block_threads}")

    if wave_specialized_tdm and num_warps < 4:
        raise ValueError(f"wave_specialized_tdm requires at least 4 waves, got {num_warps}")
    
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp

    if warp_tile_m % WMMA_M != 0:
        raise ValueError(f"warp_tile_m={warp_tile_m} must be a multiple of {WMMA_M}")
    if warp_tile_n % WMMA_N != 0:
        raise ValueError(f"warp_tile_n={warp_tile_n} must be a multiple of {WMMA_N}")
    
    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"{num_buffers}-stage buffering requires num_k_tiles >= {num_buffers}, "
            f"got {num_k_tiles} (K={K}, tile_k={tile_k})"
        )

    # ── Multi-stage pipeline schedule (compile-time) ──
    pre_loaded = num_buffers - 1
    loop_iters = (num_k_tiles - pre_loaded) // num_buffers
    _tail_start = loop_iters * num_buffers
    extra = num_k_tiles - _tail_start - pre_loaded
    tail_plan = make_tail_plan(num_buffers, pre_loaded, extra)

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep

    k_k_blocks = K // group_k
    n_k_blocks = N // group_n

    gy_compile = N // tile_n  # compile-time grid.y (== number of N-tiles)

    ###############################################
    lds_a_stride = tile_k + LDS_PAD_A_BYTES
    lds_a_stride_bytes = lds_a_stride
    lds_a_bytes  = tile_m * lds_a_stride
    
    #先不pershuffle
    lds_b_stride = tile_k + LDS_PAD_A_BYTES
    lds_b_stride_bytes = lds_b_stride
    lds_b_bytes  = tile_n * lds_b_stride

    tdm_desc_num_warps = 1 if wave_specialized_tdm else num_warps

    #####################################################
    arena_alloc = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"bmm_w8a8_{tile_m}x{tile_n}x{tile_k}"
        ),
    )

    stage_a_data_off = [0x00000, 0x04800, 0x09000, 0x0D800]
    stage_b_data_off = [0x20000, 0x29000, 0x32000, 0x3B000]

    arena_alloc.ptr = LDS_GFX1250_MAX_BYTES
    arena_total_bytes = arena_alloc.ptr
    epilogue_fence_threshold_bytes = 0

    if const_expr(use_tdm_store):
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

    if const_expr(wave_specialized_tdm):
        TDM_LOADS_PER_STEP = 1
    else:
        TDM_LOADS_PER_STEP = 2
    

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def kernel_bmm_w8a8_gfx1250(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()

        if const_expr(inst_prefetch):
            if rocdl.wave_id() == arith.constant(0, type=T.i32):
                rocdl.s_prefetch_inst_burst(num_pages=10)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bz = gpu.block_id("z")

        blk_m = bx * arith.index(tile_m)
        blk_n = by * arith.index(tile_n)

        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
        else:
            a_mcast_mask = 0
            b_mcast_mask = 0
        
        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16),
            (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1),
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.get(thr_coord, 0),
            fx.get(thr_coord, 1),
            fx.get(thr_coord, 2),
            fx.get(thr_coord, 3),
        )

        #warv m, n
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)

        m_idx = arith.index_cast(T.index, i32_m.ir_value())

        a_batch_off = bz * m_idx
        b_batch_off = bz * arith.index(N)
        c_batch_off = bz * m_idx

        ####### C resource ##########
        c_nrec = arith.index(B) * m_idx * arith.index(N * elem_bytes_d)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, num_records_bytes=c_nrec)

        ###### scale A, B############
        scale_b_total_elems = arith.index(B * n_k_blocks * k_k_blocks)
        scale_b_nrec = scale_b_total_elems * arith.index(ELEM_BYTES_SCALE)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_b_scale, num_records_bytes=scale_b_nrec)

        scale_a_total_elems = arith.index(B) * m_idx * arith.index(k_k_blocks)
        scale_a_nrec = scale_a_total_elems * arith.index(ELEM_BYTES_SCALE)
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_a_scale, num_records_bytes=scale_a_nrec)

        ############################################################################
        by_i32 = arith.index_cast(T.i32, by)
        bz_i32 = arith.index_cast(T.i32, bz)
        m_i32 = arith.index_cast(T.i32, m_idx) 

        scale_b_batch_off_i32 = bz_i32 * arith.constant(n_k_blocks * k_k_blocks, type=T.i32)
        scale_a_batch_off_i32 = bz_i32 * m_i32 * arith.constant(k_k_blocks, type=T.i32)
        scale_k_stride_i32 = arith.constant(k_k_blocks, type=T.i32)
        blk_m_i32 = arith.index_cast(T.i32, blk_m)
        warp_m_i32 = arith.index_cast(T.i32, warp_m_base)
        wave_n_index_i32 = arith.index_cast(T.i32, wave_n_idx)
        lane16_i32 = arith.index_cast(T.i32, lane16)
        b_scale_row_i32 = (by_i32 * arith.constant(tile_n // group_n, type=T.i32)
                + wave_n_index_i32 * arith.constant(warp_tile_n // group_n, type=T.i32))

        #########################################################
        def make_desc_a(memref, k_base):
            return _make_tdm_desc(
                global_ptr=arg_a,
                lds_memref=memref,
                global_offset=(blk_m + a_batch_off, k_base),
                tensor_shape=(tile_m, tile_k),
                strides=(K, 1),
                tile_shape=(tile_m, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_A_BYTES,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=a_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )
        
        def make_desc_b(memref, k_base):
            return _make_tdm_desc(
                global_ptr=arg_b,
                lds_memref=memref,
                global_offset=(blk_n + b_batch_off, k_base),
                tensor_shape=(tile_n, tile_k),
                strides=(K, 1),
                tile_shape=(tile_n, tile_k),
                elem_bytes=1,
                pad_interval=tile_k,
                pad_amount=LDS_PAD_B_BYTES,
                num_warps=tdm_desc_num_warps,
                workgroup_mask=b_mcast_mask,
                atomic_barrier_enable=atomic_barrier_enable,
                early_timeout=True,
            )
        ########################################################

        if const_expr(wave_specialized_tdm):
            tdm_wave_id = rocdl.wave_id()
            tdm_wave_is_a = arith.cmpi(arith.CmpIPredicate.eq, tdm_wave_id, arith.constant(0, type=T.i32))

            def _select_wave_tdm_value(a_value, b_value):
                return arith.select(tdm_wave_is_a, a_value, b_value)


        def _precompute_a_lane_bases(lds_ptr):
            row_base = (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
            k_half_off = lane_kgrp * arith.index(16)
            bases = []
            for wm in range_constexpr(wmma_m_rep):
                base = row_base + arith.index(wm * WMMA_M * lds_a_stride_bytes) + k_half_off
                bases.append(base)
            return lds_ptr, bases
        
        def _precompute_b_lane_bases(lds_ptr):
            row_base = (warp_n_base + lane16) * arith.index(lds_b_stride_bytes)
            k_half_off = lane_kgrp * arith.index(16)
            bases = []
            for wn in range_constexpr(wmma_n_rep):
                base = row_base + arith.index(wn * WMMA_N * lds_b_stride_bytes) + k_half_off
                bases.append(base)
            return lds_ptr, bases

        _div4 = arith.constant(2, type=T.i32)

        assert k_k_blocks == 32, (
            f"scale-row preload assumes k_k_blocks==32 (two ds-b128), got {k_k_blocks}"
        )

        def _pack_e8m0_byte(s_i8):
            s32 = fx.Int32(fx.Uint8(s_i8))
            return s32 | (s32 << 8) | (s32 << 16) | (s32 << 24)

        def _load_scale_row_vec(rsrc, byte_base_i32):
            e_lo = arith.shrui(byte_base_i32, _div4)
            v_lo = fx.Vector(buffer_ops.buffer_load(rsrc, e_lo, vec_width=4, dtype=T.i32))
            v_hi = fx.Vector(
                buffer_ops.buffer_load(
                    rsrc, e_lo + arith.constant(4, type=T.i32), vec_width=4, dtype=T.i32
                )
            )
            v8 = fx.Vector.from_elements(
                [v_lo[i] for i in range(4)] + [v_hi[i] for i in range(4)], fx.Int32
            )
            return v8.bitcast(fx.Int8)

        def _scales_for_tile(kt_idx, a_scale_rows, b_scale_row):
            a_scale_vgprs = []
            for wm in range_constexpr(wmma_m_rep):
                _row = warp_m_base + arith.index(wm * WMMA_M) + lane16
                _off = _row * arith.index(k_k_blocks) + kt_idx
                a_scale_vgprs.append(_pack_e8m0_byte(scale_a_lds.load([_off])))
            b_scale_vgpr = _pack_e8m0_byte(
                vector.extract(b_scale_row, dynamic_position=[kt_idx])
            )
            return a_scale_vgprs, b_scale_vgpr


        def _issue_frag_loads(lds_buffer, lane_base, ks):
            k_byte_off = arith.index(ks * WMMA_K)
            byte_off = lane_base + k_byte_off
            return [
                fx.Vector(lds_load_b128_raw(lds_buffer, byte_off)),
                fx.Vector(lds_load_b128_raw(lds_buffer, byte_off + arith.index(32))),
                fx.Vector(lds_load_b128_raw(lds_buffer, byte_off + arith.index(64))),
                fx.Vector(lds_load_b128_raw(lds_buffer, byte_off + arith.index(96))),
            ]

        def _assemble_frag(raw4):
            v0, v1, v2, v3 = raw4
            v01 = v0.shuffle(v1, list(range(8)))
            v23 = v2.shuffle(v3, list(range(8)))
            return v01.shuffle(v23, list(range(16)))


        def _emit_wmma(accs, wm, wn, a_frag, b_frag, a_scales, b_scales):
            idx = wm * wmma_n_rep + wn
            accs[idx] = wmma_op(
                T.vec(8, T.f32),
                b_frag,
                a_frag,
                accs[idx],
                b_scales,
                a_scales[wm],
                fmtA=0,
                fmtB=0,
                scaleAType=0,
                fmtScaleA=0,
                scaleBType=0,
                fmtScaleB=0,
            )

        def _l2_prefetch(k_base):
            if const_expr(_effective_l2_pf <= 0):
                return
            pf_k = k_base + arith.index(_effective_l2_pf * tile_k)
            tdm_ops.l2_prefetch_tile(
                arg_a,
                (blk_m + a_batch_off, pf_k),
                (tile_m, tile_k),
                (K, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )
            tdm_ops.l2_prefetch_tile(
                arg_b,
                (b_batch_off + blk_n, pf_k),
                (tile_n, tile_k),
                (K, 1),
                elem_bytes=1,
                thread_id=tx,
                block_threads=block_threads,
            )
        
        ############## pipeline ##############################

        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))
        accs = [acc_zero] * n_accs

        arena_base_ptr = arena_alloc.get_base()
        lds_a_data_f16 = lds_a_bytes // 2
        lds_b_data_f16 = lds_b_bytes // 2
        stages_a = [
            SmemPtr(arena_base_ptr, stage_a_data_off[i], T.f16, shape=(lds_a_data_f16,))
            for i in range_constexpr(num_buffers)
        ]
        stages_b = [
            SmemPtr(arena_base_ptr, stage_b_data_off[i], T.f16, shape=(lds_b_data_f16,))
            for i in range_constexpr(num_buffers) 
        ]

        stages_a_mem = [stages_a[i].get() for i in range_constexpr(num_buffers)]
        stages_b_mem = [stages_b[i].get() for i in range_constexpr(num_buffers)]
        stages_a_idx = [extract_lds_base_idx(stages_a[i]) for i in range_constexpr(num_buffers)]
        stages_b_idx = [extract_lds_base_idx(stages_b[i]) for i in range_constexpr(num_buffers)]

        a_scale_rows = None

        scale_a_lds_off = 0x12000
        scale_a_lds_bytes = tile_m * k_k_blocks  # 128 * 32 = 4096
        assert scale_a_lds_off + scale_a_lds_bytes <= stage_b_data_off[0], (
            "A-scale LDS staging overruns into the B region"
        )
        scale_a_lds = SmemPtr(arena_base_ptr, scale_a_lds_off, T.i8, shape=(scale_a_lds_bytes,))
        scale_a_lds_base_idx = extract_lds_base_idx(scale_a_lds)

        _SC_DMA_BYTES = 16
        _sc_dma_ops = scale_a_lds_bytes // (block_threads * _SC_DMA_BYTES)
        assert _sc_dma_ops * block_threads * _SC_DMA_BYTES == scale_a_lds_bytes, (
            "A-scale tile not evenly divisible by the DMA batch"
        )
        _sc_g_base = buffer_ops.extract_base_index(arg_a_scale, address_space=1)
        _sc_tile_byte = bz * m_idx * arith.index(k_k_blocks) + blk_m * arith.index(k_k_blocks)
        _sc_limit = arith.index(B) * m_idx * arith.index(k_k_blocks) - arith.index(_SC_DMA_BYTES)
        for _d in range_constexpr(_sc_dma_ops):
            _lane_byte = tx * arith.index(_SC_DMA_BYTES) + arith.index(_d * block_threads * _SC_DMA_BYTES)
            _g_off = arith.minui(_raw(_sc_tile_byte + _lane_byte), _raw(_sc_limit))
            _g_ptr = buffer_ops.create_llvm_ptr(fx.Index(_sc_g_base) + fx.Index(_g_off), address_space=1)
            _l_ptr = buffer_ops.create_llvm_ptr(fx.Index(scale_a_lds_base_idx) + _lane_byte, address_space=3)
            rocdl.cluster_load_async_to_lds(_g_ptr, _l_ptr, _SC_DMA_BYTES, offset=0, mask=None)
        rocdl.s_wait_asynccnt(0)
        workgroup_barrier(use_cluster=use_cluster)

        b_byte_base = scale_b_batch_off_i32 + b_scale_row_i32 * scale_k_stride_i32
        b_scale_row = _load_scale_row_vec(scale_b_rsrc, b_byte_base)

        _frag_ty = T.vec(16, T.i32)

        def _issue_assemble(a_buf, a_bases, b_buf, b_bases):
            a_raw = [_issue_frag_loads(a_buf, a_bases[wm], 0) for wm in range_constexpr(wmma_m_rep)]
            b_raw = [_issue_frag_loads(b_buf, b_bases[wn], 0) for wn in range_constexpr(wmma_n_rep)]
            a_frags = [_assemble_frag(r) for r in a_raw]
            b_frags = [_assemble_frag(r) for r in b_raw]
            return a_frags, b_frags

        def _issue_assemble_staggered(a_buf, a_bases, b_buf, b_bases):
            def _issue_a():
                return [_issue_frag_loads(a_buf, a_bases[wm], 0) for wm in range_constexpr(wmma_m_rep)]

            def _issue_b():
                return [_issue_frag_loads(b_buf, b_bases[wn], 0) for wn in range_constexpr(wmma_n_rep)]

            def _body(a_first):
                if const_expr(a_first):
                    a_raw = _issue_a()
                    b_raw = _issue_b()
                else:
                    b_raw = _issue_b()
                    a_raw = _issue_a()
                af = [_assemble_frag(r) for r in a_raw]
                bf = [_assemble_frag(r) for r in b_raw]
                return af + bf  # 4 a_frags + 8 b_frags

            n_frags = wmma_m_rep + wmma_n_rep
            a_first_pred = arith.cmpi(
                arith.CmpIPredicate.eq, wave_n_index_i32, arith.constant(0, type=T.i32)
            )
            if_op = scf.IfOp(a_first_pred, [_frag_ty] * n_frags, has_else=True)
            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                scf.YieldOp([_raw(f) for f in _body(True)])
            if len(if_op.regions[1].blocks) == 0:
                if_op.regions[1].blocks.append(*[])
            with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                scf.YieldOp([_raw(f) for f in _body(False)])
            frags = [fx.Vector(if_op.results[i], (16,), fx.Int32) for i in range(n_frags)]
            return frags[:wmma_m_rep], frags[wmma_m_rep:]

        DS_LOADS_PER_FRAG = 4

        def _a_streaming_pipeline(accs, a_buf, a_bases, b_buf, b_bases,
                                  a_scales, b_scale, mid_compute_callback=None):
            current_accs = list(accs)

            b_raw = [_issue_frag_loads(b_buf, b_bases[wn], 0) for wn in range_constexpr(wmma_n_rep)]
            b_frags = [_assemble_frag(r) for r in b_raw]

            a_raw = _issue_frag_loads(a_buf, a_bases[0], 0)

            for wm in range_constexpr(wmma_m_rep):
                is_last = const_expr(wm == wmma_m_rep - 1)
                if const_expr(not is_last):
                    a_raw_next = _issue_frag_loads(a_buf, a_bases[wm + 1], 0)
                    rocdl.s_wait_dscnt(DS_LOADS_PER_FRAG)
                else:
                    rocdl.s_wait_dscnt(0)

                a_frag = _assemble_frag(a_raw)

                for wn in range_constexpr(wmma_n_rep):
                    _emit_wmma(current_accs, wm, wn, a_frag, b_frags[wn], a_scales, b_scale)

                if const_expr(not is_last):
                    a_raw = a_raw_next

                if const_expr(wm == 0 and mid_compute_callback is not None):
                    rocdl.sched_barrier(0)
                    mid_compute_callback()
            return current_accs

        def compute_tile(accs_in, lds_a_idx, lds_b_idx, kt_idx, mid_compute_callback=None):
            a_buf, a_bases = _precompute_a_lane_bases(lds_a_idx)
            b_buf, b_bases = _precompute_b_lane_bases(lds_b_idx)
            a_scales, b_scale = _scales_for_tile(kt_idx, a_scale_rows, b_scale_row)
            return _a_streaming_pipeline(
                accs_in, a_buf, a_bases, b_buf, b_bases, a_scales, b_scale,
                mid_compute_callback=mid_compute_callback,
            )

        def _dg0_lane(desc, lane):
            return fx.Vector(desc.dgroup0)[lane]

        def _pack_dg0(pred, lds_addr, addr_lo, addr_hi):
            return fx.Vector.from_elements([pred, lds_addr, addr_lo, addr_hi], fx.Int32)

        stages_a_lds_addr = [
            _dg0_lane(make_desc_a(stages_a_mem[i], arith.index(0)), 1) for i in range_constexpr(num_buffers)
        ]
        stages_b_lds_addr = [
            _dg0_lane(make_desc_b(stages_b_mem[i], arith.index(0)), 1) for i in range_constexpr(num_buffers)
        ]

        desc_a_init = make_desc_a(stages_a_mem[0], arith.index(0))
        desc_b_init = make_desc_b(stages_b_mem[0], arith.index(0))
        addr_lo_a = _dg0_lane(desc_a_init, 2)
        addr_hi_a = _dg0_lane(desc_a_init, 3)
        addr_lo_b = _dg0_lane(desc_b_init, 2)
        addr_hi_b = _dg0_lane(desc_b_init, 3)
        dgroup1_a = desc_a_init.dgroup1
        dgroup1_b = desc_b_init.dgroup1

        adv_a_i32 = fx.Int32(tile_k)
        adv_b_i32 = fx.Int32(tile_k)
        pred_const = fx.Int32(1)

        def _pipeline_fence(outstanding=0):
            pipeline_fence(outstanding=outstanding, use_cluster=use_cluster)

        def _pipeline_fence_signal(outstanding=0):
            pipeline_fence_signal(outstanding=outstanding, use_cluster=use_cluster)

        def _issue_ab(load_stage, addr_box, k_prefetch=None):
            dg0_a = _pack_dg0(pred_const, stages_a_lds_addr[load_stage], addr_box[0][0], addr_hi_a)
            dg0_b = _pack_dg0(pred_const, stages_b_lds_addr[load_stage], addr_box[1][0], addr_hi_b)
            issue_tdm_loads(
                tdm_ops.TDMDescriptor2D(dg0_a, dgroup1_a),
                tdm_ops.TDMDescriptor2D(dg0_b, dgroup1_b),
            )
            addr_box[0][0] = addr_box[0][0] + adv_a_i32
            addr_box[1][0] = addr_box[1][0] + adv_b_i32
            if const_expr(k_prefetch is not None):
                _l2_prefetch(k_prefetch)

        _prologue_box = [[addr_lo_a], [addr_lo_b]]
        for i in range_constexpr(pre_loaded):
            _issue_ab(i, _prologue_box)
        addr_lo_a = _prologue_box[0][0]
        addr_lo_b = _prologue_box[1][0]

        _pipeline_fence(outstanding=TDM_LOADS_PER_STEP * (num_buffers - 2))

        _fence_outstanding = TDM_LOADS_PER_STEP * (num_buffers - 2)
        if const_expr(loop_iters > 0):
            init_args = list(accs) + [addr_lo_a, addr_lo_b]
            for loop_iter, state in range(0, loop_iters, 1, init=init_args):
                accs_in = list(state[:n_accs])
                cur_lo_a = state[n_accs]
                cur_lo_b = state[n_accs + 1]

                for buf_idx in range_constexpr(num_buffers):
                    load_stage = (buf_idx + num_buffers - 1) % num_buffers

                    _pipeline_fence_signal(outstanding=_fence_outstanding)
                    pipeline_fence_wait(use_cluster=use_cluster)

                    addr_boxes = [[cur_lo_a], [cur_lo_b]]
                    kt_idx = loop_iter * arith.index(num_buffers) + arith.index(buf_idx)

                    def _mid_tdm(
                        _ls=load_stage,
                        _ab=addr_boxes,
                        _k_off=(
                            loop_iter * arith.index(num_buffers * tile_k) + arith.index(buf_idx * tile_k)
                        ),
                    ):
                        _issue_ab(_ls, _ab, k_prefetch=_k_off)

                    rocdl.sched_barrier(0)
                    accs_in = compute_tile(
                        accs_in,
                        stages_a_idx[buf_idx],
                        stages_b_idx[buf_idx],
                        kt_idx,
                        mid_compute_callback=_mid_tdm,
                    )
                    cur_lo_a = addr_boxes[0][0]
                    cur_lo_b = addr_boxes[1][0]

                results = yield list(accs_in) + [cur_lo_a, cur_lo_b]

            accs = list(results[:n_accs])
            addr_lo_a = results[n_accs]
            addr_lo_b = results[n_accs + 1]

        if const_expr(loop_iters > 0):
            _pipeline_fence(outstanding=0)
        elif const_expr(use_cluster):
            cluster.cluster_barrier()

        _tail_had_load = False
        _tail_kt = [loop_iters * num_buffers]
        for _load_stage, _compute_stage, _outstanding in tail_plan:
            kt_const = arith.index(_tail_kt[0])
            _tail_kt[0] += 1
            if const_expr(_outstanding == -1):
                if const_expr(_tail_had_load):
                    _pipeline_fence(outstanding=0)
                rocdl.sched_barrier(0)
                accs = compute_tile(
                    accs, stages_a_idx[_compute_stage], stages_b_idx[_compute_stage], kt_const
                )
            else:
                _pipeline_fence_signal(outstanding=_outstanding)
                pipeline_fence_wait(use_cluster=use_cluster)

                _tail_mid = None
                if const_expr(_load_stage is not None):
                    _tail_had_load = True
                    _tail_box = [[addr_lo_a], [addr_lo_b]]

                    def _tail_mid_cb(_ls=_load_stage, _ab=_tail_box):
                        _issue_ab(_ls, _ab)

                    _tail_mid = _tail_mid_cb

                rocdl.sched_barrier(0)
                accs = compute_tile(
                    accs,
                    stages_a_idx[_compute_stage],
                    stages_b_idx[_compute_stage],
                    kt_const,
                    mid_compute_callback=_tail_mid,
                )
                if const_expr(_load_stage is not None):
                    addr_lo_a = _tail_box[0][0]
                    addr_lo_b = _tail_box[1][0]

        _out_elem = T.f16 if out_dtype == "f16" else (T.bf16 if out_dtype == "bf16" else None)
        _half_out = out_dtype in ("f16", "bf16")

        def epilogue_lds_stores(final_accs, d_buf, d_base):
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    imm = wm * WMMA_M * _lds_d_stride_elems + wn * _n_col_d_elems
                    store_acc_vec8_to_lds(d_buf, d_base, imm, final_accs[idx], out_elem=_out_elem)

        def epilogue_prepare_addrs():
            addrs = []
            n_stride = arith.index(N)
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    row = c_batch_off + blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
                    col_base = (blk_n + warp_n_base + arith.index(wn * WMMA_N)
                                + lane_kgrp * arith.index(8))
                    if _half_out:
                        addrs.append((row * n_stride + col_base) * arith.index(elem_bytes_d))
                    else:
                        for half in range_constexpr(2):
                            addrs.append(row * n_stride + col_base + arith.index(half * 4))
            return addrs

        def epilogue_stores(final_accs, addrs):
            addr_idx = 0
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    if _half_out:
                        addr_idx += store_acc_vec8_to_buffer(
                            final_accs[idx], c_rsrc, addrs[addr_idx],
                            out_elem=_out_elem, offset_is_bytes=True)
                    else:
                        addr_idx += store_acc_vec8_to_buffer(
                            final_accs[idx], c_rsrc, addrs[addr_idx:addr_idx + 2])

        if const_expr(use_tdm_store):
            d_lds_base_ptr = arena_base_ptr
            d_lds_f16_count = total_d_bytes // 2  # SmemPtr element type is bf16 (2B)
            d_smem = SmemPtr(d_lds_base_ptr, d_output_off, T.bf16, shape=(d_lds_f16_count,))
            d_lds_buffer = get_lds_memref(d_smem)

            warp_lds_off = (wave_m_idx * arith.index(n_warp) + wave_n_idx) \
                * arith.index(_warp_d_elems)
            d_lane_base = (warp_lds_off
                           + lane16 * arith.index(_lds_d_stride_elems)
                           + lane_kgrp * arith.index(4 * elem_bytes_d))

            wave_id_idx = arith.index_cast(T.index, rocdl.wave_id())
            d_warp_off_sgpr = wave_id_idx * arith.index(warp_d_bytes) \
                + arith.index(d_output_off)
            warp_m_off_sgpr = (wave_id_idx / arith.index(n_warp)) * arith.index(warp_tile_m)
            warp_n_off_sgpr = (wave_id_idx % arith.index(n_warp)) * arith.index(warp_tile_n)
            d_desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=arg_c,
                lds_memref=d_lds_base_ptr,
                global_offset=(c_batch_off + blk_m + warp_m_off_sgpr,
                               blk_n + warp_n_off_sgpr),
                tensor_shape=(warp_tile_m, warp_tile_n),
                strides=(N, 1),
                tile_shape=(warp_tile_m, warp_tile_n),
                elem_bytes=elem_bytes_d,
                pad_interval=warp_tile_n,
                pad_amount=LDS_PAD_D_BYTES // elem_bytes_d,
                num_warps=1,
                lds_byte_offset=d_warp_off_sgpr,
                for_store=True,
            )

            if const_expr(d_need_epilogue_fence):
                pipeline_fence(outstanding=0, use_cluster=use_cluster)
            rocdl.sched_barrier(0)

            epilogue_lds_stores(accs, d_lds_buffer, d_lane_base)
            rocdl.s_wait_dscnt(0)
            tdm_ops.tensor_store_2d(d_desc)
            tdm_ops.tensor_wait(0)
        else:
            rocdl.sched_barrier(0)
            epilogue_stores(accs, epilogue_prepare_addrs())

    cache_tag = (B, K, N, group_k, group_n, tile_m, tile_n, tile_k,
                 m_warp, n_warp, num_buffers, out_dtype, waves_per_eu,
                 l2_prefetch_distance, use_tdm_store, inst_prefetch, expert_sched_mode,
                 cluster_m, cluster_n, wave_specialized_tdm, atomic_barrier_enable)

    @flyc.jit
    def launch_bmm_w8a8_gfx1250(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_a_scale: fx.Tensor,
        arg_b_scale: fx.Tensor,
        i32_m: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            arena_alloc.finalized = False
            arena_alloc.finalize()

        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))

        launcher = kernel_bmm_w8a8_gfx1250(
            arg_c, arg_a, arg_b, arg_a_scale, arg_b_scale, i32_m
        )
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                if effective_waves_per_eu is not None:
                    _wpe = int(effective_waves_per_eu)
                    if _wpe >= 1:
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), _wpe)
                if use_cluster:
                    op.attributes["rocdl.cluster_dims"] = ir.StringAttr.get(
                        f"{cluster_m},{cluster_n},1")
        cluster_arg = (cluster_m, cluster_n, 1) if use_cluster else None
        launcher.launch(
            grid=(gx, gy_compile, B),
            block=(block_threads, 1, 1),
            cluster=cluster_arg,
            stream=stream,
        )

    llvm_opts = {}
    if expert_sched_mode:
        llvm_opts["amdgpu-expert-scheduling-mode"] = True
    if inst_prefetch:
        llvm_opts["amdgpu-inst-prefetch-distance"] = 8
    if llvm_opts:
        launch_bmm_w8a8_gfx1250.compile_hints["llvm_options"] = llvm_opts

    return launch_bmm_w8a8_gfx1250


__all__ = ["compile_bmm_w8a8_preshuffle_gfx1250"]

