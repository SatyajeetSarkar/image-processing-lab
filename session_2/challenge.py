import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Manual nearest-neighbor interpolation
def nearest_neighbor_resize(image, scale):
    height, width = image.shape

    new_height = int(height * scale)
    new_width = int(width * scale)

    result = np.zeros(
        (new_height, new_width), dtype=np.uint8
    )

    for y in range(new_height):
        for x in range(new_width):

            source_y = min(int(y / scale), height - 1)
            source_x = min(int(x / scale), width - 1)

            result[y, x] = image[source_y, source_x]

    return result

# Perform zooming and shrinking
zoom_2x = nearest_neighbor_resize(image, 2)
zoom_4x = nearest_neighbor_resize(image, 4)
shrink_half = nearest_neighbor_resize(image, 0.5)
shrink_quarter = nearest_neighbor_resize(image, 0.25)

# Display results
titles = [
    "Original Image",
    "Manual 2x Zoom",
    "Manual 4x Zoom",
    "Manual 1/2 Shrinking",
    "Manual 1/4 Shrinking"
]

images = [
    image, zoom_2x, zoom_4x,
    shrink_half, shrink_quarter
]

plt.figure(figsize=(12, 8))

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
