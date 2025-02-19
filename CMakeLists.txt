cmake_minimum_required(VERSION 3.10)
project(NNC)

add_subdirectory(src)

add_executable(NNC src/main.c)
target_include_directories(NNC PRIVATE include)

find_package(JPEG REQUIRED)
target_link_libraries(NNC PRIVATE JPEG::JPEG)
add_subdirectory(tests)