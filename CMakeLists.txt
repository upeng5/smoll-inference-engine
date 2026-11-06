cmake_minimum_required(VERSION 4.4.3)

project(ProjectCuda LANGUAGES CXX CUDA)

# Set standard for C++ and CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(CMAKE_CUDA_ARCHITECTURES native)

set(THIRD_PARTY_PATH ${CMAKE_SOURCE_DIR}/third_party)

add_subdirectory(${THIRD_PARTY_PATH}/nlohmann_json)
add_subdirectory(${THIRD_PARTY_PATH}/tokenizers-cpp EXCLUDE_FROM_ALL)

# Find CUDA toolkit package
find_package(CUDAToolkit REQUIRED)

# Gather all .cpp, .cc, and .cu files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp" 
    "src/*.cc" 
    "src/*.cu"
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

# Link CUDA runtime
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CUDA::cudart)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CUDA::cublas)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# External
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${THIRD_PARTY_PATH}/tokenizers-cpp/include)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE tokenizers_cpp)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ROOT_PATH="${CMAKE_SOURCE_DIR}")

# Optional NVTX ranges for Nsight Systems (header-only, no extra link needed)
option(ENABLE_NVTX "Emit NVTX ranges for Nsight Systems" ON)
if(ENABLE_NVTX)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENABLE_NVTX)
endif()

# Source line info in device code so nsys/ncu can map kernels to source (no perf cost)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)