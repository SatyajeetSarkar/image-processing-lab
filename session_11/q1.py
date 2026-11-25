import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# A. Simple global thresholding
_, global_thresh = cv2.threshold(
    image, 127, 255, cv2.THRESH_BINARY
)

# B. Optimal thresholding using iterative threshold selection
threshold = int(np.mean(image))

while True:
    lower = image[image <= threshold]
    upper = image[image > threshold]

    if len(lower) == 0 or len(upper) == 0:
        break

    new_threshold = int((np.mean(lower) + np.mean(upper)) / 2)

    if abs(new_threshold - threshold) < 1:
        break

    threshold = new_threshold

_, optimal_thresh = cv2.threshold(
    image, threshold, 255, cv2.THRESH_BINARY
)

print("Optimal threshold:", threshold)

# C. Adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31, 5
)

# D. Region growing from a central seed pixel
def region_growing(img, seed, tolerance=15):
    height, width = img.shape
    result = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)

    seed_y, seed_x = seed
    seed_value = int(img[seed_y, seed_x])

    queue = deque([(seed_y, seed_x)])
    visited[seed_y, seed_x] = True

    while queue:
        y, x = queue.popleft()

        if abs(int(img[y, x]) - seed_value) <= tolerance:
            result[y, x] = 255

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx

                if 0 <= ny < height and 0 <= nx < width:
                    if not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

    return result

region_result = region_growing(
    image, (image.shape[0] // 2, image.shape[1] // 2), 15
)

# Display segmentation results
titles = [
    "Original Image",
    "Global Thresholding",
    "Optimal Thresholding",
    "Adaptive Thresholding",
    "Region Growing"
]

images = [
    image, global_thresh, optimal_thresh,
    adaptive_thresh, region_result
]

plt.figure(figsize=(12, 7))

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
