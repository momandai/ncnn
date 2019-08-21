set(CMAKE_BUILD_TYPE "Release")
add_executable(benchncnn benchncnn.cpp)
target_link_libraries(benchncnn PRIVATE ncnn)
