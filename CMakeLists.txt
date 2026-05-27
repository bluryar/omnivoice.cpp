cmake_minimum_required(VERSION 3.15)

project(omnivoice-ggml-cpp
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OMNIVOICE_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml" CACHE PATH "Path to vendored ggml")

if(NOT EXISTS "${OMNIVOICE_GGML_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "vendor/ggml is missing; run git submodule update --init --recursive from ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

add_subdirectory("${OMNIVOICE_GGML_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

add_library(omnivoice_miniaudio STATIC
    vendor/miniaudio/miniaudio.c
)

target_include_directories(omnivoice_miniaudio
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor/miniaudio"
)

target_compile_definitions(omnivoice_miniaudio
    PUBLIC
        MA_NO_DEVICE_IO
        MA_NO_ENGINE
        MA_NO_GENERATION
        MA_NO_NODE_GRAPH
        MA_NO_RESOURCE_MANAGER
)

if(UNIX AND NOT APPLE)
    target_link_libraries(omnivoice_miniaudio PUBLIC m)
endif()

add_library(omnivoice_shine STATIC
    vendor/shine/src/lib/bitstream.c
    vendor/shine/src/lib/huffman.c
    vendor/shine/src/lib/l3bitstream.c
    vendor/shine/src/lib/l3loop.c
    vendor/shine/src/lib/l3mdct.c
    vendor/shine/src/lib/l3subband.c
    vendor/shine/src/lib/layer3.c
    vendor/shine/src/lib/reservoir.c
    vendor/shine/src/lib/tables.c
)

set_target_properties(omnivoice_shine PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(omnivoice_shine
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor/shine/src/lib"
)

if(UNIX AND NOT APPLE)
    target_link_libraries(omnivoice_shine PUBLIC m)
endif()

add_library(omnivoice_ggml_cpp
    src/audio_preprocess.cpp
    src/mp3.cpp
    src/omnivoice.cpp
    src/tokenizer.cpp
    src/wav.cpp
)

target_include_directories(omnivoice_ggml_cpp
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${OMNIVOICE_GGML_DIR}/include"
        "${OMNIVOICE_GGML_DIR}/src"
)

target_link_libraries(omnivoice_ggml_cpp PUBLIC ggml PRIVATE omnivoice_miniaudio omnivoice_shine)

add_library(omnivoice_nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS omnivoice_nlohmann_json)
target_include_directories(omnivoice_nlohmann_json
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/vendor"
)

add_executable(omnivoice-cli src/omnivoice_cli.cpp)
target_link_libraries(omnivoice-cli PRIVATE omnivoice_ggml_cpp)

add_executable(omnivoice-server src/omnivoice_server.cpp)
target_include_directories(omnivoice-server PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/httplib")
target_link_libraries(omnivoice-server PRIVATE omnivoice_ggml_cpp nlohmann_json::nlohmann_json)
find_package(Threads REQUIRED)
target_link_libraries(omnivoice-server PRIVATE Threads::Threads)

install(TARGETS omnivoice_ggml_cpp omnivoice-cli omnivoice-server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
