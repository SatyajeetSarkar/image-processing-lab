import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

# Check if image was loaded successfully
if image is None:
    print("Error: Could not read the image.")
    exit()

# Image dimensions
height, width = image.shape

# Number of pixels
num_pixels = image.size

# Datatype
data_type = image.dtype

# Minimum pixel intensity
min_intensity = np.min(image)

# Maximum pixel intensity
max_intensity = np.max(image)

# Mean intensity
mean_intensity = np.mean(image)

# Specified coordinate
x = 100
y = 100

# Pixel value at coordinate
if y < height and x < width:
    pixel_value = image[y, x]
else:
    pixel_value = "Coordinate out of range"

# Display information
print("Image Height:", height)
print("Image Width:", width)
print("Number of Pixels:", num_pixels)
print("Datatype:", data_type)
print("Minimum Pixel Intensity:", min_intensity)
print("Maximum Pixel Intensity:", max_intensity)
print("Mean Intensity:", mean_intensity)
print(f"Pixel Value at ({x}, {y}):", pixel_value)

# Display using OpenCV
cv2.imshow("Grayscale Image - OpenCV", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display using Matplotlib
plt.imshow(image, cmap="gray")
plt.title("Grayscale Image - Matplotlib")
plt.axis("off")
plt.show()