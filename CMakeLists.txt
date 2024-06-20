cmake_minimum_required(VERSION 3.1)
project(sobel_edge_detection)

# Find MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(sobel_edge_detection sobel_edge_detection.cpp)

# Link MPI, OpenMP, and OpenCV libraries
target_link_libraries(sobel_edge_detection 
                      ${MPI_CXX_LIBRARIES} 
                      OpenMP::OpenMP_CXX 
                      ${OpenCV_LIBS})
