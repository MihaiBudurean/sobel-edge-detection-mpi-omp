export OMP_NUM_THREADS=1
./build/sobel_edge_detection ./images/test1.jpg
export OMP_NUM_THREADS=4
./build/sobel_edge_detection_omp ./images/test1.jpg

