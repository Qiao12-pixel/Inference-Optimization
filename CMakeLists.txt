cmake_minimum_required(VERSION 3.17)
project(mini-llama CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MINI_LLAMA_CUDA "Build NVIDIA CUDA backend" OFF)

if(MINI_LLAMA_CUDA)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
    find_package(CUDAToolkit REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# --- Core library sources ---
set(MINI_LLAMA_CORE_SOURCES
    src/tensor.cc
    src/model.cc
    src/ops.cc
    src/tokenizer.cc
    src/bpe_tokenizer.cc
    src/gguf_tokenizer.cc
    src/kv_cache.cc
    src/context.cc
    src/forward.cc
    src/sampler.cc
    src/loader.cc
    src/chat.cc
    src/prompt_builder.cc
    src/terminal.cc
    src/json_vocab_tokenizer.cc
    src/batch.cc
    src/debug.cc
    src/gguf.cc
    src/gguf_loader.cc
    src/quant.cc
    src/quantized_tensor.cc
    src/thread_pool.cc
    src/matmul_optimized.cc
    src/backend.cc
    src/request_context.cc
    src/radix_tree.cc
    src/cuda_attention.cc
    src/cuda_runtime.cc
    src/cuda_tensor.cc
    src/cuda_matmul.cc
    src/cuda_quant.cc
    src/cuda_ops.cc
    src/cuda_linear_argmax.cc
    src/cuda_kv_cache.cc
)
if(MINI_LLAMA_CUDA)
    list(APPEND MINI_LLAMA_CORE_SOURCES src/cuda_attention.cu)
    list(APPEND MINI_LLAMA_CORE_SOURCES src/cuda_ops.cu)
    list(APPEND MINI_LLAMA_CORE_SOURCES src/cuda_quant.cu)
endif()

# --- Main executable ---
set(MAIN_SOURCES
    ${MINI_LLAMA_CORE_SOURCES}
    src/main.cc
)

add_executable(mini-llama ${MAIN_SOURCES})
set_target_properties(mini-llama PROPERTIES OUTPUT_NAME mini-llama)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama PRIVATE CUDA::cudart CUDA::cublas)
endif()

# --- Test executable ---
set(TEST_SOURCES
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_main.cc
    tests/test_tensor.cc
    tests/test_ops.cc
    tests/test_forward.cc
    tests/test_kv_cache.cc
    tests/test_sampler.cc
    tests/test_chat.cc
    tests/test_loader.cc
    tests/test_tokenizer.cc
    tests/test_batch.cc
    tests/test_debug.cc
    tests/test_gguf.cc
    tests/test_gguf_loader.cc
    tests/test_gguf_tokenizer.cc
    tests/test_quant.cc
    tests/test_request_context.cc
    tests/test_threaded_matmul.cc
    tests/test_radix_tree.cc
)

add_executable(mini-llama-tests ${TEST_SOURCES})
set_target_properties(mini-llama-tests PROPERTIES OUTPUT_NAME mini-llama-tests)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Enable testing
enable_testing()
add_test(NAME mini-llama-tests COMMAND mini-llama-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mini-llama-cuda-runtime-tests
    src/cuda_runtime.cc
    tests/test_cuda_runtime.cc
)
set_target_properties(mini-llama-cuda-runtime-tests PROPERTIES OUTPUT_NAME mini-llama-cuda-runtime-tests)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-cuda-runtime-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-cuda-runtime-tests PRIVATE CUDA::cudart)
endif()
add_test(NAME cuda_runtime COMMAND mini-llama-cuda-runtime-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mini-llama-cuda-matmul-tests
    src/tensor.cc
    src/cuda_runtime.cc
    src/cuda_tensor.cc
    src/cuda_matmul.cc
    tests/test_cuda_matmul.cc
)
set_target_properties(mini-llama-cuda-matmul-tests PROPERTIES OUTPUT_NAME mini-llama-cuda-matmul-tests)
if(MINI_LLAMA_CUDA)
    # cuda_matmul.cc's FP16 Tensor Core path uses the conversion kernel in
    # cuda_ops.cu. Keep the standalone matmul test link-complete as well.
    target_sources(mini-llama-cuda-matmul-tests PRIVATE src/cuda_ops.cu)
    target_compile_definitions(mini-llama-cuda-matmul-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-cuda-matmul-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()
add_test(NAME cuda_matmul COMMAND mini-llama-cuda-matmul-tests matmul WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_linear COMMAND mini-llama-cuda-matmul-tests linear WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mini-llama-cuda-quant-tests
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_cuda_quant.cc
)
set_target_properties(mini-llama-cuda-quant-tests PROPERTIES OUTPUT_NAME mini-llama-cuda-quant-tests)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-cuda-quant-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-cuda-quant-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()
add_test(NAME cuda_quant_q8_0 COMMAND mini-llama-cuda-quant-tests q8_0 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_quant_q4 COMMAND mini-llama-cuda-quant-tests q4 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_forward_quant COMMAND mini-llama-cuda-quant-tests forward WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_forward_q4 COMMAND mini-llama-cuda-quant-tests forward_q4 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mini-llama-cuda-forward-buffer-tests
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_cuda_forward_buffer.cc
)
set_target_properties(mini-llama-cuda-forward-buffer-tests PROPERTIES OUTPUT_NAME mini-llama-cuda-forward-buffer-tests)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-cuda-forward-buffer-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-cuda-forward-buffer-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()
add_test(NAME cuda_forward_buffer COMMAND mini-llama-cuda-forward-buffer-tests f32 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_forward_device_resident COMMAND mini-llama-cuda-forward-buffer-tests q8_0 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mini-llama-cuda-attention-tests
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_cuda_attention.cc
)
set_target_properties(mini-llama-cuda-attention-tests PROPERTIES OUTPUT_NAME mini-llama-cuda-attention-tests)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-cuda-attention-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-cuda-attention-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()
add_test(NAME cuda_attention COMMAND mini-llama-cuda-attention-tests direct WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_kv_attention COMMAND mini-llama-cuda-attention-tests kv WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cuda_forward_attention COMMAND mini-llama-cuda-attention-tests forward WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mini-llama-cuda-forward-linear-tests
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_cuda_forward_linear.cc
)
set_target_properties(mini-llama-cuda-forward-linear-tests PROPERTIES OUTPUT_NAME mini-llama-cuda-forward-linear-tests)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-cuda-forward-linear-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-cuda-forward-linear-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()
add_test(NAME cuda_forward_linear COMMAND mini-llama-cuda-forward-linear-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mini-llama-cuda-ops-tests
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_cuda_ops.cc
)
set_target_properties(mini-llama-cuda-ops-tests PROPERTIES OUTPUT_NAME mini-llama-cuda-ops-tests)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-cuda-ops-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-cuda-ops-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()
add_test(NAME cuda_ops COMMAND mini-llama-cuda-ops-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mini-llama-cuda-kv-cache-tests
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_cuda_kv_cache.cc
)
set_target_properties(mini-llama-cuda-kv-cache-tests PROPERTIES OUTPUT_NAME mini-llama-cuda-kv-cache-tests)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-cuda-kv-cache-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-cuda-kv-cache-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()
add_test(NAME cuda_kv_cache COMMAND mini-llama-cuda-kv-cache-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(mini-llama-cuda-forward-tests
    ${MINI_LLAMA_CORE_SOURCES}
    tests/test_cuda_forward.cc
)
set_target_properties(mini-llama-cuda-forward-tests PROPERTIES OUTPUT_NAME mini-llama-cuda-forward-tests)
if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama-cuda-forward-tests PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama-cuda-forward-tests PRIVATE CUDA::cudart CUDA::cublas)
endif()
add_test(NAME cuda_forward COMMAND mini-llama-cuda-forward-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME mvp-regression
        COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_mvp.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME golden-logits
        COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_golden.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME quant-cli-smoke
        COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_quant_cli.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME threaded-cli-smoke
        COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_threaded_cli.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME backend-cli-smoke
        COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_backend_cli.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME cuda-chat-smoke
        COMMAND ${CMAKE_COMMAND} -E env
            MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_cuda_chat_smoke.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

# --- GoogleTest target ---
option(MINI_LLAMA_BUILD_GTEST "Build optional GoogleTest-based tests" OFF)
option(MINI_LLAMA_USE_SYSTEM_GTEST "Use system-installed GoogleTest" OFF)
if(MINI_LLAMA_BUILD_GTEST)
    if(MINI_LLAMA_USE_SYSTEM_GTEST)
        find_package(GTest CONFIG QUIET HINTS /opt/homebrew/lib/cmake/GTest)
    endif()
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    set(GTEST_SOURCES
        ${MINI_LLAMA_CORE_SOURCES}
    )
    file(GLOB GTEST_TEST_FILES CONFIGURE_DEPENDS tests/gtest/*.cc)
    list(APPEND GTEST_SOURCES ${GTEST_TEST_FILES})

    add_executable(mini-llama-gtests ${GTEST_SOURCES})
    target_link_libraries(mini-llama-gtests PRIVATE GTest::gtest_main)
    set_target_properties(mini-llama-gtests PROPERTIES OUTPUT_NAME mini-llama-gtests)
    if(MINI_LLAMA_CUDA)
        target_compile_definitions(mini-llama-gtests PRIVATE MINI_LLAMA_USE_CUDA=1)
        target_link_libraries(mini-llama-gtests PRIVATE CUDA::cudart CUDA::cublas)
    endif()

    add_test(NAME mini-llama-gtests COMMAND mini-llama-gtests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# Release optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
