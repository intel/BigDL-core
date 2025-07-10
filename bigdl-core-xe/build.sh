set -e

sk_root=$(python3 -c 'import os; import skbuild; sk_root=os.path.dirname(skbuild.__file__); print(sk_root)')
torch_root=$(python3 -c 'import os; import torch; torch_root=os.path.dirname(torch.__file__); print(torch_root)')

cur_path=./bigdl-core-xe-addons
cur_build_path=${cur_path}/build

export PYTHON_EXECUTABLE=$(which python3)

cmake -GNinja -Wno-dev \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MODULE_PATH=${sk_root}/resources/cmake \
    -DCMAKE_PREFIX_PATH=${torch_root} \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_CXX_STANDARD=20 \
    -DPython_EXECUTABLE=${PYTHON_EXECUTABLE} \
    -B ${cur_build_path} ${cur_path}

cmake --build ${cur_build_path} --config Release -j

cp ${cur_build_path}/*.so .

python3 setup.py clean --all bdist_wheel --plat-name manylinux2010_x86_64 --python-tag py3