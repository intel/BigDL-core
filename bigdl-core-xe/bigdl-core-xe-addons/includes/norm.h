#pragma once

#include <torch/extension.h>

torch::Tensor rms_norm(
    torch::Tensor weight,
    torch::Tensor input,
    double eps
);

torch::Tensor layer_norm(
    torch::Tensor input,
    std::optional<torch::Tensor> weight,
    std::optional<torch::Tensor> bias,
    double eps
);
