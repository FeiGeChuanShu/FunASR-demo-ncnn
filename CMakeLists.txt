cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(FunASR-demo-ncnn)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rd/ncnn/CMakeLists.txt" OR NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rd/kaldi-native-fbank/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init\" and try again.")
endif()

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -o3 -Wall -s -fPIC -Wno-unused-function -Wno-unused-label")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -o3 -Wall -s -fPIC -Wno-unused-function -Wno-unused-label")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(BUILD_SHARED_LIBS FALSE)

set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_ROTATE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_AFFINE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_DRAWING OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_SHARED_LIB ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)

set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

add_subdirectory(3rd/ncnn)
add_subdirectory(3rd/kaldi-native-fbank)
add_subdirectory(SenseVoice)
add_subdirectory(Paraformer)
