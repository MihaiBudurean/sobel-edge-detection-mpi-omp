#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <mpi.h>

namespace fs = std::filesystem;

// Parallel Sobel Edge Detection using MPI
// Parallel Sobel Edge Detection using MPI
void sobel_edge_detection_mpi(const cv::Mat& src, cv::Mat& dst, int rank, int size)
{
    // Get the number of rows and columns in the source image
    int rows = src.rows;
    int cols = src.cols;

    // Calculate the number of rows to process for each process
    int local_rows = rows / size;

    // Determine the starting and ending rows for the current process
    int start_row = rank * local_rows;
    int end_row = (rank == size - 1) ? rows : start_row + local_rows;

    // Extract the portion of the image that the current process will handle
    cv::Mat local_src = src.rowRange(start_row, end_row);

    // Create matrices to store the gradients in x and y directions for the local portion
    cv::Mat grad_x, grad_y;

    // Compute the gradient in the x direction using the Sobel operator for the local portion
    cv::Sobel(local_src, grad_x, CV_32F, 1, 0);

    // Compute the gradient in the y direction using the Sobel operator for the local portion
    cv::Sobel(local_src, grad_y, CV_32F, 0, 1);

    // Initialize the local destination image with the same width and the calculated local height
    cv::Mat local_dst(local_rows, cols, CV_32F);

    // Iterate over each pixel in the local portion of the image
    for (int y = 0; y < local_rows; ++y)
        for (int x = 0; x < cols; ++x)
        {
            // Get the gradient values at the current pixel in x and y directions
            float gx = grad_x.at<float>(y, x);
            float gy = grad_y.at<float>(y, x);

            // Compute the magnitude of the gradient vector using the Euclidean norm
            local_dst.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
        }

    // Convert the floating-point gradient magnitude image to an 8-bit image
    local_dst.convertTo(local_dst, CV_8U);

    // The root process initializes the destination image with the full size
    if (rank == 0)
        dst = cv::Mat(rows, cols, CV_8U);

    // Gather the processed local images from all processes into the destination image on the root process
    MPI_Gather(local_dst.data, local_rows * cols, MPI_UNSIGNED_CHAR,
               dst.data, local_rows * cols, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2)
    {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;

        MPI_Finalize();
        return -1;
    }

    fs::path image_path = argv[1];
    cv::Mat image;

    if (rank == 0)
    {
        image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

        if (image.empty())
        {
            std::cerr << "Error: Could not open or find the image " << image_path << std::endl;
            MPI_Finalize();
            return -1;
        }
    }

    int rows, cols;
    if (rank == 0)
    {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        image.create(rows, cols, CV_8U);
    
    MPI_Bcast(image.data, rows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    cv::Mat edge_image;

    // Parallel Sobel Edge Detection using MPI
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = MPI_Wtime();
    sobel_edge_detection_mpi(image, edge_image, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = MPI_Wtime();

    if (rank == 0)
    {
        double duration = stop - start;
        std::cout << "Sequential Sobel Edge Detection Time: " << duration << " seconds" << std::endl;
        std::string relative_path =  image_path.string();
        std::string relative_path_without_extension = relative_path.substr(0, relative_path.find_last_of("."));
        cv::imwrite(relative_path_without_extension + "_edge_mpi.jpg", edge_image);
    }

    MPI_Finalize();
    return 0;
}
