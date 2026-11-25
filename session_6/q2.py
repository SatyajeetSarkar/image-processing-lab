import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Define Roberts kernels
roberts_x = np.array([[1, 0],
                      [0, -1]], dtype=np.float32)

roberts_y = np.array([[0, 1],
                      [-1, 0]], dtype=np.float32)

# Apply Roberts operator
rx = cv2.filter2D(image, cv2.CV_32F, roberts_x)
ry = cv2.filter2D(image, cv2.CV_32F, roberts_y)
roberts = cv2.convertScaleAbs(cv2.magnitude(rx, ry))

# Define Prewitt kernels
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float32)

# Apply Prewitt operator
px = cv2.filter2D(image, cv2.CV_32F, prewitt_x)
py = cv2.filter2D(image, cv2.CV_32F, prewitt_y)
prewitt = cv2.convertScaleAbs(cv2.magnitude(px, py))

# Apply Sobel operator
sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))

# Apply Laplacian operator
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Apply Canny edge detector
canny = cv2.Canny(image, 100, 200)

# Display results
images = [image, roberts, prewitt, sobel, laplacian, canny]
titles = ["Original", "Roberts", "Prewitt", "Sobel", "Laplacian", "Canny"]

plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
