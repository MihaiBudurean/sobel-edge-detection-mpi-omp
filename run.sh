# Execute sequential algorithm
export OMP_NUM_THREADS=1
./build/sobel_edge_detection ./images/test1.jpg

# Execute OpenMP algorithm
export OMP_NUM_THREADS=4
./build/sobel_edge_detection_omp ./images/test1.jpg

# Execute MPI algorithm
mpirun -np 4 ./build/sobel_edge_detection_mpi ./images/test1.jpg
