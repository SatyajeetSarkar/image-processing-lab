import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("coins.png")

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# A. Image preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# B. Thresholding
_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# C. Morphological noise removal
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, kernel, iterations=2
)

# Find sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# D. Distance transformation
dist_transform = cv2.distanceTransform(
    opening, cv2.DIST_L2, 5
)

# E. Foreground marker generation
_, sure_fg = cv2.threshold(
    dist_transform,
    0.45 * dist_transform.max(),
    255, 0
)

sure_fg = np.uint8(sure_fg)

# F. Unknown region generation
unknown = cv2.subtract(sure_bg, sure_fg)

# Generate markers
num_labels, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so background becomes 1
markers = markers + 1

# Mark unknown regions as zero
markers[unknown == 255] = 0

# G. Watershed transformation
result = image.copy()
markers = cv2.watershed(result, markers)

# H. Boundary detection
result[markers == -1] = [0, 0, 255]

# Count detected objects (exclude background and watershed boundaries)
object_count = len(np.unique(markers)) - 2

print("Number of detected objects:", object_count)



# Display results
titles = [
    "Original Image",
    "Thresholding",
    "Morphological Opening",
    "Distance Transform",
    "Foreground Markers",
    "Watershed Result"
]

images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    thresh,
    opening,
    dist_transform,
    sure_fg,
    cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(12, 8))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray" if i != 5 and i != 0 else None)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
