#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
namespace ch = std::chrono;

void sobel_edge_detection(const cv::Mat& src, cv::Mat& dst)
{
    // Get the number of rows and columns in the source image
    int rows = src.rows;
    int cols = src.cols;

    // Create matrices to store the gradients in x and y directions
    cv::Mat grad_x, grad_y;

    // Compute the gradient in the x direction using the Sobel operator
    cv::Sobel(src, grad_x, CV_32F, 1, 0);

    // Compute the gradient in the y direction using the Sobel operator
    cv::Sobel(src, grad_y, CV_32F, 0, 1);

    // Initialize the destination image with the same size as the source image
    // and set its type to CV_32F to hold floating-point numbers
    dst = cv::Mat(rows, cols, CV_32F);

    // Iterate over each pixel in the image
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
        {
            // Get the gradient values at the current pixel in x and y directions
            float gx = grad_x.at<float>(y, x);
            float gy = grad_y.at<float>(y, x);

            // Compute the magnitude of the gradient vector using the Euclidean norm
            dst.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
        }

    // Convert the floating-point gradient magnitude image to an 8-bit image
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

    auto start = ch::high_resolution_clock::now();
    sobel_edge_detection(image, edge_image);
    auto stop = ch::high_resolution_clock::now();

    ch::duration<double> duration = stop - start;

    std::cout << "Sequential Sobel Edge Detection Time: " << duration.count() << " seconds" << std::endl;
    
    std::string relative_path =  image_path.string();
    std::string relative_path_without_extension = relative_path.substr(0, relative_path.find_last_of("."));
    cv::imwrite(relative_path_without_extension + "_edge_seq.jpg", edge_image);

    return 0;
}
