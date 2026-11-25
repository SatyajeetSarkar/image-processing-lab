import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Calculate image information
height, width = image.shape
total_pixels = image.size
datatype = image.dtype
minimum = np.min(image)
maximum = np.max(image)
mean = np.mean(image)

# Display image information
print("Image Height:", height)
print("Image Width:", width)
print("Total Pixels:", total_pixels)
print("Data Type:", datatype)
print("Minimum Intensity:", minimum)
print("Maximum Intensity:", maximum)
print("Mean Intensity:", mean)

# Display pixel value at a specified coordinate
x, y = 100, 100
print(f"Pixel value at ({x}, {y}):", image[y, x])

# Display using OpenCV
cv2.imshow("Grayscale Image - OpenCV", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display using Matplotlib
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Grayscale Image - Matplotlib")
plt.axis("off")
plt.show()
