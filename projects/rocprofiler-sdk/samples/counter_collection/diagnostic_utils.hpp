// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Diagnostic utilities for tracking down GPU memory faults in counter collection
// Issue: kernel-replay-counters test experiences random GPU memory access faults on MI325
//

#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <sys/utsname.h>

namespace rocprofiler_diag {

// Check if diagnostic mode is enabled via environment variable
inline bool is_diagnostic_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char* env = std::getenv("ROCPROF_DIAGNOSTIC");
        enabled = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
    }
    return enabled == 1;
}

// Print GPU device information
inline void print_gpu_info(int dev_id = 0) {
    if (!is_diagnostic_enabled()) return;

    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, dev_id);
    if (err != hipSuccess) {
        fprintf(stderr, "[DIAG] Failed to get device properties: %s\n", hipGetErrorString(err));
        return;
    }

    fprintf(stderr, "\n[DIAG] ========== GPU Device Info (Device %d) ==========\n", dev_id);
    fprintf(stderr, "[DIAG] Name: %s\n", prop.name);
    fprintf(stderr, "[DIAG] gcnArchName: %s\n", prop.gcnArchName);
    fprintf(stderr, "[DIAG] Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "[DIAG] Shared Memory Per Block: %zu bytes\n", prop.sharedMemPerBlock);
    fprintf(stderr, "[DIAG] Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    fprintf(stderr, "[DIAG] Multiprocessor Count: %d\n", prop.multiProcessorCount);
    fprintf(stderr, "[DIAG] Compute Capability: %d.%d\n", prop.major, prop.minor);
    fprintf(stderr, "[DIAG] PCI Bus ID: %d:%d.%d\n", prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
    fprintf(stderr, "[DIAG] ================================================\n\n");
}

// Print system information
inline void print_system_info() {
    if (!is_diagnostic_enabled()) return;

    struct utsname buf;
    if (uname(&buf) == 0) {
        fprintf(stderr, "\n[DIAG] ========== System Info ==========\n");
        fprintf(stderr, "[DIAG] OS: %s\n", buf.sysname);
        fprintf(stderr, "[DIAG] Kernel: %s\n", buf.release);
        fprintf(stderr, "[DIAG] Machine: %s\n", buf.machine);
        fprintf(stderr, "[DIAG] ===================================\n\n");
    }
}

// Print memory allocation info
inline void print_memory_alloc(const char* name, void* ptr, size_t size) {
    if (!is_diagnostic_enabled()) return;

    fprintf(stderr, "[DIAG] Memory allocated: %s\n", name);
    fprintf(stderr, "[DIAG]   Address: %p\n", ptr);
    fprintf(stderr, "[DIAG]   Size: %zu bytes (%.2f MB)\n", size, size / (1024.0 * 1024.0));

    // Check if pointer is accessible
    hipPointerAttribute_t attr;
    hipError_t err = hipPointerGetAttributes(&attr, ptr);
    if (err == hipSuccess) {
        fprintf(stderr, "[DIAG]   Memory Type: ");
        switch(attr.type) {
            case hipMemoryTypeHost: fprintf(stderr, "Host\n"); break;
            case hipMemoryTypeDevice: fprintf(stderr, "Device\n"); break;
            case hipMemoryTypeManaged: fprintf(stderr, "Managed\n"); break;
            default: fprintf(stderr, "Unknown (%d)\n", attr.type); break;
        }
        fprintf(stderr, "[DIAG]   Device ID: %d\n", attr.device);
    } else {
        fprintf(stderr, "[DIAG]   WARNING: Could not get pointer attributes: %s\n", hipGetErrorString(err));
    }
}

// Print buffer info before kernel launch
inline void print_pre_kernel_state(const char* kernel_name, dim3 grid, dim3 block) {
    if (!is_diagnostic_enabled()) return;

    fprintf(stderr, "\n[DIAG] ========== Pre-Kernel Launch ==========\n");
    fprintf(stderr, "[DIAG] Kernel: %s\n", kernel_name);
    fprintf(stderr, "[DIAG] Grid: (%u, %u, %u)\n", grid.x, grid.y, grid.z);
    fprintf(stderr, "[DIAG] Block: (%u, %u, %u)\n", block.x, block.y, block.z);
    fprintf(stderr, "[DIAG] Total threads: %u\n", grid.x * grid.y * grid.z * block.x * block.y * block.z);

    // Get free and total memory
    size_t free_mem, total_mem;
    hipError_t err = hipMemGetInfo(&free_mem, &total_mem);
    if (err == hipSuccess) {
        fprintf(stderr, "[DIAG] GPU Memory: %.2f MB free / %.2f MB total (%.1f%% used)\n",
                free_mem / (1024.0 * 1024.0),
                total_mem / (1024.0 * 1024.0),
                100.0 * (1.0 - (double)free_mem / total_mem));
    }

    fprintf(stderr, "[DIAG] ==========================================\n\n");
}

// Check for GPU errors after operation
inline bool check_gpu_error(const char* operation) {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "\n[DIAG] !!! GPU ERROR after %s !!!\n", operation);
        fprintf(stderr, "[DIAG] Error: %s (%d)\n", hipGetErrorString(err), err);
        fprintf(stderr, "[DIAG] This may indicate the crash location!\n\n");
        return true;
    }
    return false;
}

// Synchronize and check for errors
inline void sync_and_check(const char* checkpoint) {
    if (!is_diagnostic_enabled()) return;

    fprintf(stderr, "[DIAG] Checkpoint: %s - synchronizing...\n", checkpoint);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "[DIAG] !!! SYNC FAILED at %s !!!\n", checkpoint);
        fprintf(stderr, "[DIAG] Error: %s (%d)\n", hipGetErrorString(err), err);
        fprintf(stderr, "[DIAG] ** This is likely where the crash occurred **\n\n");
    } else {
        fprintf(stderr, "[DIAG] Checkpoint: %s - OK\n", checkpoint);
    }
}

// Print counter collection start
inline void print_counter_collection_start(size_t num_counters) {
    if (!is_diagnostic_enabled()) return;

    fprintf(stderr, "\n[DIAG] ========== Starting Counter Collection ==========\n");
    fprintf(stderr, "[DIAG] Number of counters: %zu\n", num_counters);
    fprintf(stderr, "[DIAG] Timestamp: %ld\n", std::chrono::system_clock::now().time_since_epoch().count());
    fprintf(stderr, "[DIAG] ==================================================\n\n");
}

// Print summary at end
inline void print_diagnostic_summary() {
    if (!is_diagnostic_enabled()) return;

    fprintf(stderr, "\n[DIAG] ========== Test Completed Successfully ==========\n");
    fprintf(stderr, "[DIAG] If this message appears, no GPU memory fault occurred\n");
    fprintf(stderr, "[DIAG] ===================================================\n\n");
}

} // namespace rocprofiler_diag
