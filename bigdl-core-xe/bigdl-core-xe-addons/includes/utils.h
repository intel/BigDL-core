#pragma once

#include <functional>
#include <torch/extension.h>

#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
#include <c10/xpu/XPUStream.h>
#else
#include <ipex.h>
#endif

namespace utils {
    static inline sycl::queue& get_queue(const torch::Device& device) {
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
        return c10::xpu::getCurrentXPUStream(device.index()).queue();
#else
        c10::impl::VirtualGuardImpl impl(device.type());
        c10::Stream c10_stream = impl.getStream(c10::Device(device));
        return xpu::get_queue_from_stream(c10_stream);
#endif
    }

    static inline sycl::event submit_kernel(std::function<void(sycl::handler&)> kernel, const at::Device& device, const char * desc) {
        sycl::queue& queue = get_queue(device);
        sycl::event event = queue.submit(kernel);
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 3
        // xpu::profiler_record(desc, event);
#else
        xpu::profiler_record(desc, event);
#endif
        return event;
    }
}
