#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from setuptools import setup
import torch
import intel_extension_for_pytorch as ipex
from torch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension

VERSION = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
               'version.txt'), 'r').read().strip()
include_dir = str(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "includes"))
_ipex_version = ipex.__version__
_major_version = ''.join(_ipex_version.split('.')[:2])  # equal to 21
suffix_name = "-" + _major_version

setup(
    name="bigdl-core-llm" + suffix_name,
    version=VERSION,
    ext_modules=[
        DPCPPExtension('bigdl_core_llm', [
            'xpu_addon_ops.cpp',
            'norm.cpp',
        ],
        extra_compile_args=["-std=c++20"],
        include_dirs=[include_dir])
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension.with_options(use_ninja=False)
    })
