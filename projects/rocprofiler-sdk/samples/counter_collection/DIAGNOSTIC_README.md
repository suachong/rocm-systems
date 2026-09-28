# Diagnostic Changes for GPU Memory Fault Investigation

## Problem

The `kernel-replay-counters` test experiences random GPU memory access faults on MI325 (gfx942) hardware:
- Failures are intermittent (not every run)
- Different tests fail each time (`kernel-replay-basic`, `kernel-replay-counters`)
- GPU memory faults at random addresses (e.g., `0x7e0a60412000`, `0x7551d8402000`)
- Only affects MI325, not MI300X (both gfx942)

## Diagnostic Changes

This PR adds comprehensive diagnostic instrumentation to help identify the root cause.

### Files Changed

1. **diagnostic_utils.hpp** (NEW)
   - Utility functions for printing GPU state, memory allocations, and checkpoints
   - Only active when `ROCPROF_DIAGNOSTIC=1` environment variable is set

2. **main.cpp** (MODIFIED)
   - Added diagnostic calls at key points:
     - System and GPU info at startup
     - Memory allocation tracking
     - Pre-kernel launch state
     - Post-operation GPU error checking
     - Synchronization checkpoints

3. **CMakeLists.txt** (MODIFIED)
   - Enabled `ROCPROF_DIAGNOSTIC=1` for counter collection tests
   - Added `RESOURCE_LOCK` to prevent parallel execution (reduces contention)

### How to Use

#### Running Locally

```bash
# Enable diagnostics
export ROCPROF_DIAGNOSTIC=1

# Run the test
cd build/samples/counter_collection
./counter-collection-device-profiling
```

#### CI Testing

The diagnostic mode is **automatically enabled** for CI runs. Check the CI logs for:

```
[DIAG] ========== GPU Device Info ==========
[DIAG] Name: ...
[DIAG] gcnArchName: ...
[DIAG] ================================================
```

### What to Look For in Logs

When a crash occurs, the diagnostics will show **the last successful checkpoint**, helping narrow down the crash location:

#### Example: Crash During Kernel Launch

```
[DIAG] Pre-Kernel Launch: kernelC
[DIAG] Grid: (512, 1, 1)
[DIAG] GPU Memory: 45123.45 MB free / 65536.00 MB total
[DIAG] ==========================================
Memory access fault by GPU node-4 on address 0x7551d8402000
```

→ **Crash happened during kernelC execution**

#### Example: Crash During Memory Allocation

```
[DIAG] Memory allocated: C_d
[DIAG]   Address: 0x7f8a40000000
[DIAG]   Size: 1048576 bytes (1.00 MB)
[DIAG] Checkpoint: after A_d memcpy - OK
Memory access fault by GPU node-4 on address 0x7551d8402000
```

→ **Crash happened after memcpy but before next checkpoint**

### Expected Diagnostic Output

On a **successful run**, you should see:

```
[DIAG] ========== System Info ==========
[DIAG] OS: Linux
[DIAG] Kernel: 6.18.33.2-microsoft-standard-WSL2
[DIAG] ===================================

[DIAG] ========== GPU Device Info (Device 0) ==========
[DIAG] Name: AMD Radeon Graphics
[DIAG] gcnArchName: gfx942
[DIAG] Total Global Memory: 64.00 GB
[DIAG] ================================================

[DIAG] Checkpoint: after hipSetDevice - synchronizing...
[DIAG] Checkpoint: after hipSetDevice - OK

[DIAG] Memory allocated: gpuMem
[DIAG]   Address: 0x7f8a40000000
[DIAG]   Size: 4 bytes
[DIAG]   Memory Type: Device
[DIAG]   Device ID: 0

[DIAG] ========== Starting Counter Collection ==========
[DIAG] Number of counters: 2
[DIAG] ==================================================

[DIAG] ========== Pre-Kernel Launch ==========
[DIAG] Kernel: kernelA
[DIAG] Grid: (1, 1, 1)
[DIAG] Block: (1, 1, 1)
[DIAG] GPU Memory: 65500.00 MB free / 65536.00 MB total
[DIAG] ==========================================

[DIAG] Checkpoint: after all kernelA/B launches - synchronizing...
[DIAG] Checkpoint: after all kernelA/B launches - OK

[DIAG] ========== Test Completed Successfully ==========
[DIAG] If this message appears, no GPU memory fault occurred
[DIAG] ===================================================
```

## What We're Testing

### Hypotheses to Validate

1. **Buffer size issue**: Are memory allocations reported correctly? Check the sizes in diagnostic output.

2. **Race condition**: Does adding `RESOURCE_LOCK` (serializing tests) prevent crashes?

3. **Timing issue**: Which checkpoint is the LAST one before crash? This tells us where the fault occurs.

4. **Hardware difference**: Compare MI325 vs MI300X diagnostic output - any differences in memory layout or GPU state?

### Data to Collect

From multiple CI runs (both passing and failing), collect:

1. **GPU device name and gcnArchName** - confirm which exact hardware
2. **Last successful checkpoint** - narrows crash location
3. **Memory allocation addresses** - look for patterns in bad addresses
4. **GPU memory usage** - is memory exhaustion a factor?
5. **Test parallelism** - did RESOURCE_LOCK help?

## Next Steps Based on Findings

### If crash location identified:
- Add more fine-grained diagnostics around that specific operation
- Check rocprofiler-sdk buffer allocation at that point
- Add explicit GPU synchronization before that operation

### If RESOURCE_LOCK helps:
- Root cause is hardware contention or race condition in rocprofiler-sdk global state
- Solution: Keep resource lock, investigate rocprofiler-sdk thread safety

### If memory addresses show pattern:
- E.g., all addresses in same range → GPU memory allocator bug
- E.g., addresses look like host pointers → host/device memory confusion

### If no diagnostic output appears:
- Crash happens before diagnostic code runs
- Problem is in rocprofiler-sdk initialization, not sample code

## Reverting These Changes

These changes are **non-invasive**:
- Zero overhead when `ROCPROF_DIAGNOSTIC=0` (default for local builds)
- Only active in CI or when explicitly enabled
- No functional changes to the actual test logic
- Can be removed after root cause is found

To disable temporarily:
```cmake
# In CMakeLists.txt, change:
"${counter-collection-functional-counter-env};ROCPROF_DIAGNOSTIC=1"
# To:
"${counter-collection-functional-counter-env};ROCPROF_DIAGNOSTIC=0"
```

## References

- Original issue: Intermittent GPU memory faults in PR #12085 CI runs
- Similar pattern: Issue #11436 (kfd_events crashes on gfx942)
- ROCprofiler-SDK docs mention firmware version mismatches cause similar faults
