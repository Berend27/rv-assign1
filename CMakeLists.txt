cmake_minimum_required(VERSION 3.16)
project(assign1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED PATHS /usr/local/lib/cmake/opencv4 NO_DEFAULT_PATH)

add_executable(assign1_task1
    code/assign1_task1.cpp
)

add_executable(assign1_task2
    code/assign1_task2.cpp
)

add_executable(assign1_task3
    code/assign1_task3.cpp
)

target_include_directories(assign1_task3 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(assign1_task3 PRIVATE ${OpenCV_LIBS})

set_target_properties(assign1_task3 PROPERTIES
    BUILD_RPATH "/usr/local/lib"
    INSTALL_RPATH "/usr/local/lib"
)

target_include_directories(assign1_task1 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(assign1_task1 PRIVATE ${OpenCV_LIBS})

target_include_directories(assign1_task2 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(assign1_task2 PRIVATE ${OpenCV_LIBS})

set_target_properties(assign1_task1 PROPERTIES
    BUILD_RPATH "/usr/local/lib"
    INSTALL_RPATH "/usr/local/lib"
)

set_target_properties(assign1_task2 PROPERTIES
    BUILD_RPATH "/usr/local/lib"
    INSTALL_RPATH "/usr/local/lib"
)