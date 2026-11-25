import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.png")

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 1.5)

# A. Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# B. Edge linking using morphological closing
kernel = np.ones((3, 3), np.uint8)
linked_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# C. Hough Line Transform
line_result = image.copy()

lines = cv2.HoughLinesP(
    linked_edges,
    1,
    np.pi / 180,
    threshold=80,
    minLineLength=40,
    maxLineGap=10
)

if lines is not None:
    lines = np.asarray(lines).reshape(-1, 4)

    for x1, y1, x2, y2 in lines:
        cv2.line(line_result, (x1, y1), (x2, y2),
                 (0, 0, 255), 2)

# D. Hough Circle Transform
circle_result = image.copy()
circles = cv2.HoughCircles(
    blur, cv2.HOUGH_GRADIENT,
    dp=1.2, minDist=30,
    param1=100, param2=35,
    minRadius=10, maxRadius=150
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0]:
        cv2.circle(circle_result, (x, y), r, (0, 255, 0), 2)
        cv2.circle(circle_result, (x, y), 2, (0, 0, 255), 3)

# Display results
titles = ["Original Image", "Canny Edges",
          "Edge Linking", "Hough Lines", "Hough Circles"]

images = [image, edges, linked_edges, line_result, circle_result]

plt.figure(figsize=(13, 8))

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
               if i in [0, 3, 4] else images[i], cmap=None if i in [0, 3, 4] else "gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

print("Number of detected lines:",
      0 if lines is None else len(lines))

print("Number of detected circles:",
      0 if circles is None else len(circles[0]))
