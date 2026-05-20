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

add_library(omnivoice_ggml_cpp
    src/audio_preprocess.cpp
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

target_link_libraries(omnivoice_ggml_cpp PUBLIC ggml)

add_executable(omnivoice-cli src/omnivoice_cli.cpp)
target_link_libraries(omnivoice-cli PRIVATE omnivoice_ggml_cpp)

add_executable(omnivoice-server src/omnivoice_server.cpp)
target_include_directories(omnivoice-server PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor/httplib")
target_link_libraries(omnivoice-server PRIVATE omnivoice_ggml_cpp)
find_package(Threads REQUIRED)
target_link_libraries(omnivoice-server PRIVATE Threads::Threads)

install(TARGETS omnivoice_ggml_cpp omnivoice-cli omnivoice-server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
