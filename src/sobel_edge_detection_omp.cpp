#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <omp.h>

namespace fs = std::filesystem;

// Parallel Sobel Edge Detection using OpenMP
void sobel_edge_detection_omp(const cv::Mat& src, cv::Mat& dst)
{
    int rows = src.rows;
    int cols = src.cols;

    cv::Mat grad_x, grad_y;
    cv::Sobel(src, grad_x, CV_32F, 1, 0);
    cv::Sobel(src, grad_y, CV_32F, 0, 1);

    dst = cv::Mat(rows, cols, CV_32F);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
        {
            float gx = grad_x.at<float>(y, x);
            float gy = grad_y.at<float>(y, x);
            dst.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
        }

    dst.convertTo(dst, CV_8U);
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    fs::path image_path = argv[1];
    cv::Mat image;

    image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cerr << "Error: Could not open or find the image " << image_path << std::endl;
        return -1;
    }

    int rows = image.rows;
    int cols = image.cols;

    image.create(rows, cols, CV_8U);

    cv::Mat edge_image;

    double start = omp_get_wtime();
    sobel_edge_detection_omp(image, edge_image);
    double stop = omp_get_wtime();

    double duration = stop - start;

    std::cout << "OpenMP Sobel Edge Detection Time: " << duration << " seconds" << std::endl;
    
    std::string relative_path =  image_path.string();
    std::string relative_path_without_extension = relative_path.substr(0, relative_path.find_last_of("."));
    cv::imwrite(relative_path_without_extension + "_edge_omp.jpg", edge_image);

    return 0;
}
