cmake_minimum_required(VERSION 3.12)
project(llm C CXX)


if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    # ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if (MSVC)
    if (ONLYAVX)
        # Enable AVX
        add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
    else()
        # Enable AVX2
        add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        if (AVXVNNI)
            add_compile_options(/D__AVXVNNI__)
        endif()
    endif()
    # Disable Warnings
    add_compile_options(/W0)
    # Others
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
    if (ONLYAVX)
        add_compile_options(-fPIC -mavx)
    elseif(ONLYAVX2)
        add_compile_options(-fPIC -march=haswell -mtune=haswell)
    else()
        add_compile_options(-fPIC -march=native -mtune=native)
    endif()
endif()

add_library(quantize SHARED quantize.h quantize.c)
if (MSVC)
    target_link_libraries(quantize kernel32)
else()
    set_target_properties(quantize PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
    target_link_libraries(quantize -ldl)
endif()