# Execute sequential algorithm
export OMP_NUM_THREADS=1
./build/sobel_edge_detection ./images/test3.jpg

# Execute OpenMP algorithm
export OMP_NUM_THREADS=8
./build/sobel_edge_detection_omp ./images/test3.jpg

# Execute MPI algorithm
mpirun -np 3 ./build/sobel_edge_detection_mpi ./images/test3.jpg
