#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <mpi.h>

namespace fs = std::filesystem;

// Parallel Sobel Edge Detection using MPI
void sobel_edge_detection_mpi(const cv::Mat& src, cv::Mat& dst, int rank, int size)
{
    int rows = src.rows;
    int cols = src.cols;
    int local_rows = rows / size;
    int start_row = rank * local_rows;
    int end_row = (rank == size - 1) ? rows : start_row + local_rows;

    cv::Mat local_src = src.rowRange(start_row, end_row);
    cv::Mat grad_x, grad_y;
    cv::Sobel(local_src, grad_x, CV_32F, 1, 0);
    cv::Sobel(local_src, grad_y, CV_32F, 0, 1);

    cv::Mat local_dst(local_rows, cols, CV_32F);

    for (int y = 0; y < local_rows; ++y)
        for (int x = 0; x < cols; ++x)
        {
            float gx = grad_x.at<float>(y, x);
            float gy = grad_y.at<float>(y, x);
            local_dst.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
        }

    local_dst.convertTo(local_dst, CV_8U);

    if (rank == 0)
        dst = cv::Mat(rows, cols, CV_8U);

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
