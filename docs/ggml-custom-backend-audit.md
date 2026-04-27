# GGML Custom Backend Audit

## Baseline

- Standalone project GGML: `vendor/ggml`, commit `49f84a924f6ea4fc2ef73dbbd8cc4d734b54bd6d`, tag `v0.9.11`, cloned from `https://github.com/ggml-org/ggml.git`.
- Workbench comparison source: the original `ggbond` root `vendor/ggml` tree contained the CUDA backend experiments/fixes summarized below.
- Comparison target: standalone project `vendor/ggml` against upstream `v0.9.11`.
- Standalone project policy: keep `vendor/ggml` as a clean upstream submodule
  and carry only the required OmniVoice CUDA fixes in
  `patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch`.

## Public API Result

There are no custom public GGML operators.

- No added `GGML_OP_*` enum values.
- No added public `GGML_API ggml_*` functions.
- No modified public headers were detected in this comparison.

The current customizations are CUDA backend internals only. The standalone
project does not commit these customizations directly into `vendor/ggml`; its
patch file contains only the `IM2COL` launch layout change and the
`CONV_TRANSPOSE_1D` batch/index fix required by the verified CUDA decode path.

## Historical Workbench Modified Files

The historical GGML workbench tree differed from upstream in these files. These
experimental changes are not committed into this standalone project's
`vendor/ggml` submodule.

- `src/ggml-cuda/ggml-cuda.cu`
- `src/ggml-cuda/softmax.cu`
- `src/ggml-cuda/top-k.cu`
- `src/ggml-cuda/im2col.cu`
- `src/ggml-cuda/conv-transpose-1d.cu`

Standalone `vendor/ggml` is kept clean against upstream.
`patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch` modifies:

- `src/ggml-cuda/im2col.cu`
- `src/ggml-cuda/conv-transpose-1d.cu`

## CUDA Backend Customizations

### Non-FA Attention Core Fusion

File: `src/ggml-cuda/ggml-cuda.cu`

This adds backend-only recognition and optional fusion for:

```text
MUL_MAT(k, q) -> SOFT_MAX(masked, scale, max_bias=0) -> MUL_MAT(v, softmax)
```

It introduces:

- `ggml_cuda_non_fa_attention_core_match`
- graph pattern matching for `MUL_MAT -> SOFT_MAX -> MUL_MAT`
- an explicit F32 attention core kernel
- a tiled F32 attention core kernel
- graph-level execution replacement when enabled

Environment gates:

- `GGML_CUDA_FUSE_NON_FA_ATTENTION_CORE`
- `GGML_CUDA_NON_FA_ATTENTION_CORE_KERNEL`
- `GGML_CUDA_NON_FA_ATTENTION_CORE_TILED`
- `GGML_CUDA_NON_FA_ATTENTION_CORE_TILED_KEY_WARPS`
- `GGML_CUDA_NON_FA_ATTENTION_CORE_TILED_MAX_KV`
- `GGML_CUDA_NON_FA_ATTENTION_CORE_TILED_STATS`
- `GGML_CUDA_DEBUG_NON_FA_ATTENTION_CORE`

Status:

- Backend-internal experimental fusion.
- Not a new GGML op.
- Default-off unless env gates are set.

### Small Masked Softmax Kernel

File: `src/ggml-cuda/softmax.cu`

This adds a narrow F32 masked softmax path for small row widths.

Environment gates:

- `GGML_CUDA_MASKED_SOFT_MAX_SMALL`
- `GGML_CUDA_MASKED_SOFT_MAX_SMALL_WARPS`
- `GGML_CUDA_DEBUG_MASKED_SOFT_MAX_SMALL`

Preconditions include:

- `src0`, `src1`, and `dst` are F32.
- mask exists and `src2` is null.
- `max_bias == 0`.
- row width is `<= WARP_SIZE`.
- `src0` and `dst` are contiguous.

Status:

- Backend-internal kernel override for `GGML_OP_SOFT_MAX`.
- Not a new GGML op.
- Default-off unless env gate is set.

### Simple TOP_K Kernel

File: `src/ggml-cuda/top-k.cu`

This adds a simple deterministic F32/I32 top-k CUDA kernel selected by:

- `GGML_CUDA_TOP_K_SIMPLE`

Status:

- Backend-internal alternate implementation for `GGML_OP_TOP_K`.
- Not a new GGML op.
- Default-off unless env gate is set.

### IM2COL Launch Layout Change

File: `src/ggml-cuda/im2col.cu`

This changes the CUDA launch/indexing layout:

- upstream uses a 2-D grid dimension for output width
- current tree folds output width into `blockIdx.x`

Status:

- Backend implementation change for existing im2col behavior.
- No public op change.

### Conv Transpose 1D Batch/Index Fix

File: `src/ggml-cuda/conv-transpose-1d.cu`

This changes indexing in the CUDA kernel:

- separates `idx`, `out_index`, and `batch_idx`
- accounts for `src1_ne1` batch/channel stride in input offset
- iterates kernel positions directly instead of scanning all input positions

Status:

- Backend implementation fix/change for existing `GGML_OP_CONV_TRANSPOSE_1D`.
- No public op change.

## Conclusion

The current `vendor/ggml` custom work is not a set of custom GGML C API operators. It is a set of CUDA backend experiments/fixes layered under existing GGML ops:

- attention fusion under existing `MUL_MAT`, `SOFT_MAX`, `MUL_MAT`
- small masked softmax under existing `SOFT_MAX`
- simple top-k under existing `TOP_K`
- CUDA implementation changes for `IM2COL` and `CONV_TRANSPOSE_1D`

For the pure C++ OmniVoice runtime, the stable path avoids the default-off
attention, softmax, and top-k experiments from the root workbench. The standalone
runtime applies only the `IM2COL` and `CONV_TRANSPOSE_1D` CUDA fixes from
`patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch` when building the verified
long-text CUDA decode path. These fixes should eventually be upstreamed or moved
to a project-owned GGML fork.

## Reproduction Commands

Run these commands from the standalone repository root after initializing the
submodule.

```bash
git -C vendor/ggml rev-parse HEAD
git -C vendor/ggml diff --stat

patch_file="$PWD/patches/ggml-v0.9.11-omnivoice-cuda-fixes.patch"
git -C vendor/ggml apply --check "$patch_file"

rg --no-filename -o "GGML_OP_[A-Z0-9_]+" vendor/ggml/include/ggml.h vendor/ggml/src/ggml.c | sort -u
rg --no-filename -o "GGML_API[^;{]+ggml_[A-Za-z0-9_]+" vendor/ggml/include | sed 's/[[:space:]]\+/ /g' | sort -u
```
