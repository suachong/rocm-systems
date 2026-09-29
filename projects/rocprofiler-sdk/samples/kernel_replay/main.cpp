// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/utsname.h>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(                                                                               \
                stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_err), __FILE__, __LINE__); \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while(0)

// Diagnostic helper functions
namespace diag {
inline bool is_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char* env = std::getenv("ROCPROF_DIAGNOSTIC");
        enabled = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
    }
    return enabled == 1;
}

inline void print_system_info() {
    if (!is_enabled()) return;
    struct utsname buf;
    if (uname(&buf) == 0) {
        fprintf(stderr, "\n[DIAG] ========== System Info ==========\n");
        fprintf(stderr, "[DIAG] OS: %s\n", buf.sysname);
        fprintf(stderr, "[DIAG] Kernel: %s\n", buf.release);
        fprintf(stderr, "[DIAG] Machine: %s\n", buf.machine);
        fprintf(stderr, "[DIAG] ===================================\n\n");
    }
}

inline void print_gpu_info(int dev_id = 0) {
    if (!is_enabled()) return;
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
    fprintf(stderr, "[DIAG] ================================================\n\n");
}

inline void print_memory_alloc(const char* name, void* ptr, size_t size) {
    if (!is_enabled()) return;
    fprintf(stderr, "[DIAG] Memory allocated: %s\n", name);
    fprintf(stderr, "[DIAG]   Address: %p\n", ptr);
    fprintf(stderr, "[DIAG]   Size: %zu bytes\n", size);

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
    }
}

inline void checkpoint(const char* msg) {
    if (!is_enabled()) return;
    fprintf(stderr, "[DIAG] Checkpoint: %s\n", msg);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "[DIAG] !!! Sync FAILED at %s: %s\n", msg, hipGetErrorString(err));
    }
}

inline void summary() {
    if (!is_enabled()) return;
    fprintf(stderr, "\n[DIAG] ========== Test Completed Successfully ==========\n");
    fprintf(stderr, "[DIAG] No GPU memory faults occurred\n");
    fprintf(stderr, "[DIAG] ===================================================\n\n");
}
}  // namespace diag

// Replayed kernel: distinctive block size 67. Busy enough for host-trap PC sampling.
constexpr int kReplayBlock = 67;
// Opt-out kernel: distinctive block size 64. Tools leave replay_pass_count NULL for this dispatch.
constexpr int kOptOutBlock = 64;

__global__ void
bump(int* x)
{
    volatile unsigned acc = 0;
    for(int i = 0; i < 64 * 1024; ++i)
        acc += static_cast<unsigned>(i + threadIdx.x);
    if(threadIdx.x == 0 && acc != ~0u) atomicAdd(x, 1);
}

__global__ void
nudge(int* x)
{
    if(threadIdx.x == 0) atomicAdd(x, 1);
}

int
main()
{
    diag::print_system_info();
    diag::print_gpu_info(0);

    // Workaround for MI325 memory access fault: Add explicit device selection and sync
    // to ensure GPU context is fully initialized before profiler intercepts queues
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipDeviceSynchronize());

    int* replayed = nullptr;
    int* opted    = nullptr;

    diag::checkpoint("before hipMalloc");
    HIP_CHECK(hipMalloc(&replayed, sizeof(int)));
    diag::print_memory_alloc("replayed", replayed, sizeof(int));

    HIP_CHECK(hipMalloc(&opted, sizeof(int)));
    diag::print_memory_alloc("opted", opted, sizeof(int));

    diag::checkpoint("after hipMalloc");

    // Ensure GPU can access these buffers before profiler instruments them
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemset(replayed, 0, sizeof(int)));
    HIP_CHECK(hipMemset(opted, 0, sizeof(int)));
    diag::checkpoint("after hipMemset");

    // Additional sync after memset to ensure memory is ready
    HIP_CHECK(hipDeviceSynchronize());

    if(diag::is_enabled()) {
        fprintf(stderr, "[DIAG] Launching bump kernel (replayed, block=%d)\n", kReplayBlock);
    }
    bump<<<1, kReplayBlock>>>(replayed);
    HIP_CHECK(hipGetLastError());
    diag::checkpoint("after bump kernel");

    if(diag::is_enabled()) {
        fprintf(stderr, "[DIAG] Launching nudge kernel (opt-out, block=%d)\n", kOptOutBlock);
    }
    nudge<<<1, kOptOutBlock>>>(opted);
    HIP_CHECK(hipGetLastError());
    diag::checkpoint("after nudge kernel");

    HIP_CHECK(hipDeviceSynchronize());
    diag::checkpoint("after final sync");

    int replayed_h = 0;
    int opted_h    = 0;
    HIP_CHECK(hipMemcpy(&replayed_h, replayed, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&opted_h, opted, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(replayed));
    HIP_CHECK(hipFree(opted));

    // Ensure complete cleanup before test exit to free GPU resources
    HIP_CHECK(hipDeviceSynchronize());

    printf("[app] replayed_bump=%d opted_nudge=%d\n", replayed_h, opted_h);
    if(replayed_h != 1)
    {
        fprintf(stderr,
                "[app] FAIL: replayed_bump=%d (expected 1; snapshot/restore did not isolate "
                "passes)\n",
                replayed_h);
        return EXIT_FAILURE;
    }
    if(opted_h != 1)
    {
        fprintf(stderr, "[app] FAIL: opted_nudge=%d (expected 1)\n", opted_h);
        return EXIT_FAILURE;
    }

    diag::summary();
    return EXIT_SUCCESS;
}
