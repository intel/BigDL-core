cmake_minimum_required(VERSION 3.16)
project(bigdl_core LANGUAGES C CXX)

find_package(IntelSYCL REQUIRED)
find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)
find_library(TORCH_PYTHON_LIBRARYS torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

set(MODULE_NAME bigdl_core)

set(SYCL_SRCS  norm.cpp)

set(ALL_SRCS ${SYCL_SRCS}
             xpu_addon_ops.cpp)

if(WIN32)
    add_compile_options(-Wno-ignored-attributes)
    add_python_library(${MODULE_NAME} SHARED SOURCES ${ALL_SRCS})
    python_extension_module(${MODULE_NAME})
    set_target_properties(${MODULE_NAME} PROPERTIES SUFFIX ${PYTHON_EXTENSION_MODULE_SUFFIX})
else()
    add_python_extension(${MODULE_NAME} SOURCES ${ALL_SRCS})
endif()

add_compile_definitions(TORCH_EXTENSION_NAME=${MODULE_NAME})
add_sycl_to_target(TARGET ${MODULE_NAME} SOURCES ${SYCL_SRCS})
target_include_directories(${MODULE_NAME} PRIVATE includes)
target_include_directories(${MODULE_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${MODULE_NAME} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARYS})
