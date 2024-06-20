import cv2
import os

def upscale_image_opencv(input_path, output_dir, scale_factor):
    # Read the image
    img = cv2.imread(input_path)
    # Calculate the new size
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    # Resize the image
    upscaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    # Create output path
    output_path = os.path.join(output_dir, 'test3.jpg')
    # Save the image
    cv2.imwrite(output_path, upscaled_img)
    print(f"Image upscaled using OpenCV and saved to {output_path}")

# Example usage
input_image_path = './images/test2.jpg'
output_image_dir = './images'
scale_factor = 2.0  # Scale factor (2.0 means double the size)

upscale_image_opencv(input_image_path, output_image_dir, scale_factor)
