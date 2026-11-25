import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Define a high-pass filter kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Apply high-pass filtering
high_pass = cv2.filter2D(image, cv2.CV_32F, kernel)
high_pass = cv2.convertScaleAbs(high_pass)

# Apply Laplacian filtering
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Apply Gaussian blur for unsharp masking
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Subtract blurred image from original
mask = cv2.subtract(image, blurred)

# Apply unsharp masking
unsharp = cv2.add(image, mask)

# Apply high-boost filtering
A = 1.5
high_boost = cv2.addWeighted(image, A, blurred, 1 - A, 0)

# Display results
images = [image, high_pass, laplacian, unsharp, high_boost]
titles = ["Original", "High-pass", "Laplacian", "Unsharp Masking", "High-boost"]

plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
