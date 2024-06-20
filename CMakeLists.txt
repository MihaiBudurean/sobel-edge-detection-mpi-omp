cmake_minimum_required(VERSION 3.1)
project(sobel_edge_detection)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(OpenMP REQUIRED)
# find_package(MPI REQUIRED)

# Include OpenCV and MPI libraries
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executables
add_executable(sobel_edge_detection src/sobel_edge_detection.cpp)
add_executable(sobel_edge_detection_omp src/sobel_edge_detection_omp.cpp)
# add_executable(sobel_edge_detection_mpi sobel_edge_detection_mpi.cpp)

# Link MPI, OpenMP, and OpenCV libraries
target_link_libraries(sobel_edge_detection ${OpenCV_LIBS})
target_link_libraries(sobel_edge_detection_omp ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
# target_link_libraries(sobel_edge_detection_mpi ${OpenCV_LIBS}  ${MPI_CXX_LIBRARIES})
