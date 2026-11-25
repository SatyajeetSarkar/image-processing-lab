import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Define masks for point and line detection
point_mask = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

horizontal_mask = np.array([[-1, -1, -1],
                            [ 2,  2,  2],
                            [-1, -1, -1]])

vertical_mask = np.array([[-1,  2, -1],
                          [-1,  2, -1],
                          [-1,  2, -1]])

diagonal_mask = np.array([[-1, -1,  2],
                          [-1,  2, -1],
                          [ 2, -1, -1]])

# Apply masks
point_response = cv2.filter2D(image, cv2.CV_32F, point_mask)
horizontal_response = cv2.filter2D(image, cv2.CV_32F, horizontal_mask)
vertical_response = cv2.filter2D(image, cv2.CV_32F, vertical_mask)
diagonal_response = cv2.filter2D(image, cv2.CV_32F, diagonal_mask)

# Threshold the responses
point = np.uint8(np.abs(point_response) > 180)
horizontal = np.uint8(np.abs(horizontal_response) > 180)
vertical = np.uint8(np.abs(vertical_response) > 180)
diagonal = np.uint8(np.abs(diagonal_response) > 180)

# Convert original image to color
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Overlay detected features
result[point > 0] = [0, 0, 255]
result[horizontal > 0] = [0, 255, 0]
result[vertical > 0] = [255, 0, 0]
result[diagonal > 0] = [0, 255, 255]

# Display results
titles = ["Original Image", "Point Detection",
          "Horizontal Lines", "Vertical Lines",
          "Diagonal Lines", "Combined Detection"]

images = [image, point, horizontal, vertical, diagonal, result]

plt.figure(figsize=(12, 8))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray" if i < 5 else None)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
