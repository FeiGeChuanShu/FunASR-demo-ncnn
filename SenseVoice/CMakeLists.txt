include_directories(${CMAKE_SOURCE_DIR}/3rd/ncnn/src)
include_directories(${CMAKE_SOURCE_DIR}/3rd/kaldi-native-fbank)

set(src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/src/punct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensevoice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fbank_feat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)
add_executable(sensevoice_demo ${src_files})
target_include_directories(sensevoice_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sensevoice_demo PRIVATE ncnn kaldi-native-fbank-core)
