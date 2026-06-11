cmake_minimum_required(VERSION 3.16)
project(zkresnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /W3)
else()
    add_compile_options(-O2 -Wall -Wextra -march=native)
endif()

add_executable(zkresnet
    src/main.cpp
    src/prover.cpp
    src/verifier.cpp
)

target_include_directories(zkresnet PRIVATE src)
