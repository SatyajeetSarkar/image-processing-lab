import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Image acquisition
image = cv2.imread("coins.png")

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# 2. Grayscale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Noise removal and enhancement
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Edge detection
edges = cv2.Canny(blur, 50, 150)

# 5. Otsu thresholding
_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 6. Morphological noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, kernel, iterations=2
)

# 7. Background and foreground detection
sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

_, sure_fg = cv2.threshold(
    dist, 0.45 * dist.max(), 255, 0
)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 8. Generate markers
num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 9. Watershed segmentation
result = image.copy()
markers = cv2.watershed(result, markers)

# 10. Detect boundaries and count objects
result[markers == -1] = [0, 0, 255]

object_count = len(np.unique(markers[markers > 1]))

print("Total objects detected:", object_count)

# 11. Resize image
resized = cv2.resize(
    result, None, fx=0.5, fy=0.5,
    interpolation=cv2.INTER_AREA
)

# 12. Compress image
cv2.imwrite(
    "pipeline_compressed.jpg",
    resized,
    [cv2.IMWRITE_JPEG_QUALITY, 70]
)

print("Original size:", os.path.getsize("coins.png"), "bytes")
print("Compressed size:",
      os.path.getsize("pipeline_compressed.jpg"), "bytes")

# Display pipeline results
titles = [
    "Original Image",
    "Grayscale",
    "Canny Edges",
    "Otsu Threshold",
    "Watershed Result",
    "Resized and Compressed"
]

images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    gray,
    edges,
    thresh,
    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(cv2.imread("pipeline_compressed.jpg"),
                 cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(13, 8))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray" if i in [1, 2, 3] else None)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
