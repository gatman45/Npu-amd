# XDNA2 NPU Runtime — Empirical Stability Findings on Strix Point (HX 370)

**Author:** Gatman
**Date:** 2026-04-13  
**Hardware:** Acer Nitro 16S IA — AMD Ryzen AI 9 HX 370 (Strix Point), DDR5 5600 MT/s  
**Driver:** xdna-driver 7.0.3.57  
**Runtime:** XRT (xrt_coreutil.dll) — black-box profiling via read-only proxy layer  

---

## Summary

This report documents empirical findings on XDNA2 NPU runtime behavior during LLM inference on Strix Point (HX 370), obtained through black-box profiling of the XRT runtime.

Key findings:
- A descriptor padding threshold of ~152 bytes is required for stable execution of larger models (4B–9B)
- Memory flag `0x17` and 64KB alignment for large buffers correlate strongly with stability
- Previously observed `WinError 10054` / `WinError 10061` failures are fully resolved under this configuration
- Performance gains of up to **+70% decode throughput** and **-39% total inference time** observed with `--pmode performance`
- Gain pattern strongly suggests INT4 weights undergo intermediate conversion (→ INT8 or FP16) with RAM roundtrip before NPU execution

---

## Motivation

The XDNA2 NPU in Ryzen AI 300 series is marketed at 50 TOPS (INT8). However, practical LLM inference using FLM (flm.exe) on Strix Point exhibited:

- Systematic crashes on models ≥4B parameters (`WinError 10054`, `WinError 10061`)
- No available documentation explaining the failure mode
- No public runtime stability thresholds for descriptor padding or buffer alignment

This investigation was conducted to identify the root cause empirically.

---

## Methodology

A **read-only proxy layer** was placed over `xrt_coreutil.dll` to observe:
- Buffer Object (BO) allocation patterns and memory flags
- Descriptor submission structures
- Scheduling and dispatch behavior during token-by-token inference

Cross-referenced sources (all open source):
- [xdna-driver](https://github.com/amd/xdna-driver) — descriptor / submission structures
- [XRT (Xilinx/XRT)](https://github.com/Xilinx/XRT) — BO allocation + memory flags (`xrt_mem.h`)
- [llvm-project (amd fork)](https://github.com/amd/llvm-project) — implied buffer/layout constraints
- [mlir-aie](https://github.com/Xilinx/mlir-aie) — AIE tiling and data movement model
- [Triton-XDNA](https://github.com/amd/Triton-XDNA) — kernel → runtime mapping behavior
- [IRON architecture docs](https://github.com/amd/IRON) — high-level data movement model

---

## Performance Results

### Baseline vs `--pmode performance`

| Model | Decode baseline (t/s) | Decode optimized (t/s) | Gain | Total time baseline | Total time optimized | Time reduction |
|---|---|---|---|---|---|---|
| lfm2:1.2b | 29.95 | 50.99 | **+70%** | 29.7s | 18.0s | **-39%** |
| qwen3.5:2b | 18.07 | 24.29 | **+34%** | — | — | — |
| qwen3.5:4b | 9.37 | 12.83 | **+37%** | — | — | — |
| qwen3.5:9b | 5.68 | 7.68 | **+35%** | 156.1s | 120.4s | **-23%** |
| deepseek-r1:8b | 7.59 | 10.75 | **+42%** | — | — | — |

TTFT (Time To First Token) reduced **8–20%** across all models.

### Interpretation

Gains are **strongest on smaller models** (lfm2:1.2b: +70%) and decrease with model size. This pattern is characteristic of **overhead-bound workloads**, not compute-bound ones.

This strongly suggests that `--pmode performance` reduces scheduling latency and dispatch overhead — not raw compute throughput. The dominant overhead on smaller models is the **runtime dispatch pipeline**, not matrix multiply time.

---

## Stability Findings

### Problem

Models ≥4B parameters failed consistently under default configuration:
- `WinError 10054` — connection reset during NPU submission
- `WinError 10061` — connection refused on dispatch

These errors originate at the XRT user-space / kernel driver interface, not at the compute level. The NPU was not saturated — the failure was in descriptor submission.

### Solution — Empirical Stability Thresholds

The following configuration eliminates both error classes in my environment:

| Parameter | Value | Notes |
|---|---|---|
| Descriptor padding | ~152 bytes | Stable region within ~48–192B range |
| Memory flags | `0x17` | Applied to large buffer allocations |
| Buffer alignment | 64KB | Enforced for buffers above threshold size |

**No instability observed** across 50+ inference runs after applying this configuration.

### Root Cause Hypothesis

The Strix Point (XDNA2) driver appears to share significant codebase heritage with the Phoenix / Hawk Point (XDNA gen1) driver. The descriptor padding constraints and alignment requirements suggest that XDNA2-specific buffer handling is **not fully abstracted** from the gen1 submission path.

This is consistent with the observed behavior:
- Small models (≤2B) rarely hit the failure threshold — their descriptor footprint stays within the gen1-compatible range
- Larger models (≥4B) systematically exceed it without explicit padding

---

## INT4 Execution Path — Observations

### Expected vs Observed

The XDNA2 NPU theoretically supports INT4 natively. However, observed behavior suggests that in the current runtime (driver 7.0.3.57, FLM inference path):

- **INT4 weights are not executed natively end-to-end**
- An intermediate conversion step occurs (INT4 → INT8 or FP16) before NPU dispatch
- Converted tensors appear to transit through **system RAM** before execution

### Evidence

1. The +70% decode gain on lfm2:1.2b under `--pmode performance` is disproportionate for a compute-bound workload — it indicates the bottleneck was in the dispatch/conversion pipeline
2. RAM bandwidth (5600 MT/s DDR5) appears as a measurable constraint even on small models where NPU compute should dominate
3. Buffer allocation patterns observed via proxy show intermediate-precision allocations inconsistent with pure INT4 end-to-end execution

### Implication

The **50 TOPS figure** marketed for XDNA2 NPU is theoretical INT8 performance. Under current runtime conditions for generalist LLM inference:
- Active TOPS are significantly lower due to conversion overhead
- System RAM bandwidth becomes an artificial bottleneck
- Gains from `--pmode performance` are largely gains recovered from this conversion overhead, not from increased NPU utilization

---

## Recommendations for Maintainers / Contributors

1. **Descriptor padding documentation** — publish the valid padding range for XDNA2 submission descriptors, particularly for workloads exceeding 2B parameter models
2. **INT4 native execution path** — clarify whether end-to-end INT4 execution is available in the current runtime, and if not, document the conversion path and its RAM bandwidth implications
3. **Driver heritage** — document which components of the XDNA2 driver are shared with XDNA gen1, and what the implications are for buffer alignment requirements
4. **Stability thresholds** — expose or document the padding/alignment constraints so application developers can configure them explicitly rather than discovering them empirically

---

## Reproducibility

Configuration tested on:
- **Hardware:** Acer Nitro 16S IA, HX 370, 32GB DDR5 5600 MT/s
- **Driver:** xdna-driver 7.0.3.57
- **Runtime:** XRT / xrt_coreutil.dll (proxy observation, read-only)
- **Inference engine:** FLM (flm.exe)
- **Models tested:** lfm2:1.2b, qwen3.5:2b, qwen3.5:4b, qwen3.5:9b, deepseek-r1:8b

Logs and minimal reproduction available on request.

---

## Related Issues / Discussions

- [llama.cpp #9181 — NPU Support Feature Request](https://github.com/ggml-org/llama.cpp/issues/9181)
- [llama.cpp #14377 — Official AMD Ryzen AI NPU Support](https://github.com/ggml-org/llama.cpp/issues/14377)
- [xdna-driver](https://github.com/amd/xdna-driver)
- [Triton-XDNA](https://github.com/amd/Triton-XDNA)
- [mlir-aie](https://github.com/Xilinx/mlir-aie)

---

*This report is shared as exploratory empirical research. All observations are from a single hardware configuration and should be independently verified. Feedback and reproduction attempts welcome.*
 


