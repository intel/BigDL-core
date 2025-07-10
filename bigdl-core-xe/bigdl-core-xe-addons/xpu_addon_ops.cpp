//
// Copyright 2016 The BigDL Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//


#include <torch/extension.h>
#include <vector>

#include "norm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layer_norm", &layer_norm, "fused layer norm");
}

#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 6

TORCH_LIBRARY_FRAGMENT(ipex_llm, m) {
    m.def("layer_norm(Tensor input, Tensor? weight, Tensor? bias, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(ipex_llm, XPU, m) {
    m.impl("layer_norm", &layer_norm);
}

#endif
