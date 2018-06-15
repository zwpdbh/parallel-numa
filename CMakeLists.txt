cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_COMPILER "/home/wzhao/gcc-4.9.4/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/wzhao/gcc-4.9.4/bin/g++")

#set (CMAKE_CXX_FLAGS "-O2")

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR})

include_directories("/usr/local/include/")
include_directories("/usr/include")
include_directories("/home/wzhao/gcc-4.9.4/include/c++/4.9.4/")

link_directories("/home/wzhao/gcc-4.9.4/lib64")
# command-line options: 
# sdc 5 16 32 0 0 features/_DSC0%s.jpg features/gps-sift-small.csv
add_executable(xiaoxin_knn
  main.cpp
  main.h
  common.h
  flann_kdtree_index.h
  flann_kmeans_index.h
  flann_linear_index.h
  csv_parser.cpp
  csv_parser.h
  csv_parser.cpp
  rbc_index.cpp
  rbc_index.h
  sdc_index.cpp
  sdc_index.h)
# specify target compile options
target_link_libraries(xiaoxin_knn PUBLIC flann gsl gslcblas rbc gomp)
target_compile_options(xiaoxin_knn PUBLIC "-fopenmp" "-O2")


# two demo show the read and write on local and remote node situations
add_executable(
  serial_access
  serial_access.cpp
)
target_link_libraries(serial_access PUBLIC pthread numa)
target_compile_options(serial_access PUBLIC "-pthread" "-O2")

add_executable(
  matrix_multiplication
  matrix_multiplication.cpp
)
target_link_libraries(matrix_multiplication PUBLIC pthread numa)
target_compile_options(matrix_multiplication PUBLIC "-pthread" "-O2" "-mhard-float")

add_executable(
  openmp_matrix_multiplication
  openmp_matrix_multiplication.cpp
)
target_link_libraries(openmp_matrix_multiplication numa gomp)
target_compile_options(openmp_matrix_multiplication PUBLIC "-fopenmp" "-O2")


add_executable(
  parallel
  parallel/main.cpp
  parallel/mylib.h
  parallel/mylib.cpp
  parallel/pthread_parallel.h
  parallel/pthread_parallel.cpp
  parallel/openmp_parallel.h
  parallel/openmp_parallel.cpp
)
target_link_libraries(parallel numa gomp pthread)
target_compile_options(parallel PUBLIC "-fopenmp" "-O2" "-pthread" "-mhard-float")