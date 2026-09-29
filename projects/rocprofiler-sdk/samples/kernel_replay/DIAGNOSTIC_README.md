# Diagnostic Changes for kernel-replay GPU Memory Faults

## Problem

The `kernel-replay-counters` test crashes with GPU memory access faults on MI325:

```
Memory access fault by GPU node-2 on address 0x7343be000000
Reason: Write access to a read-only page
```

**Key warning before crash:**
```
[queue-intercept] device-memory ring-buffer requested but profiling requires 
system-memory InterceptQueue; falling back to system-memory ring
```

This indicates rocprofiler is forcing system-memory buffers but the GPU is trying to write to what appears as a read-only page.

## Diagnostic Changes

### Files Modified

1. **main.cpp** - Added comprehensive diagnostic functions:
   - `diag::is_enabled()` - Check `ROCPROF_DIAGNOSTIC` env var
   - `diag::print_system_info()` - OS/kernel information
   - `diag::print_gpu_info()` - GPU device properties
   - `diag::print_memory_alloc()` - Memory allocation details and type
   - `diag::checkpoint()` - Synchronization checkpoints with error checking
   - `diag::summary()` - Success marker

2. **CMakeLists.txt** - Enabled diagnostics and added resource lock:
   - Added `ROCPROF_DIAGNOSTIC=1` to `kernel-replay-counters` environment
   - Added `RESOURCE_LOCK "rocprofiler_kernel_replay"` to serialize all kernel-replay tests

## How to Use

### In CI

Diagnostics are **automatically enabled** for `kernel-replay-counters`. Check CI logs for:

```
[DIAG] ========== GPU Device Info ==========
[DIAG] Name: ...
[DIAG] gcnArchName: gfx942
```

### Locally

```bash
export ROCPROF_DIAGNOSTIC=1
cd build/samples/kernel_replay
./kernel-replay-counters
```

## What to Look For

### Successful Run

```
[DIAG] ========== System Info ==========
[DIAG] OS: Linux
[DIAG] Kernel: ...
[DIAG] ========== GPU Device Info (Device 0) ==========
[DIAG] Name: AMD Radeon Graphics
[DIAG] gcnArchName: gfx942
[DIAG] Checkpoint: before hipMalloc
[DIAG] Memory allocated: replayed
[DIAG]   Address: 0x...
[DIAG]   Size: 4 bytes
[DIAG]   Memory Type: Device
[DIAG] Memory allocated: opted
[DIAG]   Address: 0x...
[DIAG]   Size: 4 bytes
[DIAG]   Memory Type: Device
[DIAG] Checkpoint: after hipMalloc
[DIAG] Checkpoint: after hipMemset
[DIAG] Launching bump kernel (replayed, block=67)
[DIAG] Checkpoint: after bump kernel
[DIAG] Launching nudge kernel (opt-out, block=64)
[DIAG] Checkpoint: after nudge kernel
[DIAG] Checkpoint: after final sync
[DIAG] ========== Test Completed Successfully ==========
```

### Crash Scenario

The last checkpoint before the crash shows where the fault occurred:

```
[DIAG] Launching bump kernel (replayed, block=67)
Memory access fault by GPU node-2 on address 0x7343be000000
```

→ **Crash happened during/after bump kernel with counter collection**

## Analysis

### Memory Type Mismatch

The warning indicates a fundamental issue:
1. Application requests device-memory buffers (normal for GPU kernels)
2. Rocprofiler counter collection forces system-memory buffers
3. GPU tries to write to these buffers
4. System-memory buffers may be mapped read-only for the GPU
5. GPU faults with "Write access to a read-only page"

### Why Only MI325?

- Different memory architecture or mapping between MI300X and MI325
- MI325 firmware/driver may enforce stricter memory protections
- System-memory mapping behavior differs between GPU generations

## Potential Fixes

### Option 1: Fix ROCprofiler Memory Mapping

In `queue_controller.cpp`, ensure system-memory buffers are mapped with write permissions for GPU:

```cpp
// When forcing system-memory ring, ensure GPU can write
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED | 
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
```

### Option 2: Allow Device-Memory for Counter Collection

Investigate if counter collection can use device-memory buffers instead of forcing system-memory:

```cpp
// In counters_client.cpp or queue_controller.cpp
// Check if device-memory ring can be preserved for counter collection
```

### Option 3: Add Explicit Memory Synchronization

Before counter collection starts, ensure GPU-visible memory:

```cpp
// In counters_client.cpp tool_init()
// Ensure counter buffers are properly mapped for GPU access
hsa_amd_agents_allow_access(...);
```

## Testing Strategy

1. **With RESOURCE_LOCK**: Does serializing tests prevent the crash? (tests race condition hypothesis)
2. **Check memory attributes**: Diagnostic output shows if memory type changes during profiling
3. **Compare MI300X vs MI325**: Run same diagnostic on both to see memory mapping differences

## References

- Issue #12408: Intermittent GPU memory faults
- queue_controller.cpp: System-memory fallback implementation
- Known limitation: Priority and CU-mask not preserved in system-memory mode
