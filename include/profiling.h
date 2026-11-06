#pragma once

// Optional NVTX ranges for Nsight Systems timelines.
// Enable with: cmake -DENABLE_NVTX=ON ...  (compiles to nothing otherwise)
#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>

struct NvtxRange
{
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange() { end(); }
    // Close the range before the end of scope (safe to call more than once).
    void end()
    {
        if (active)
        {
            nvtxRangePop();
            active = false;
        }
    }
    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;

private:
    bool active = true;
};
#else
struct NvtxRange
{
    explicit NvtxRange(const char *) {}
    void end() {}
};
#endif

#define NVTX_CONCAT_INNER(a, b) a##b
#define NVTX_CONCAT(a, b) NVTX_CONCAT_INNER(a, b)
#define NVTX_RANGE(name) NvtxRange NVTX_CONCAT(nvtx_range_, __LINE__)(name)
