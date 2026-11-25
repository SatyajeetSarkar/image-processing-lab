import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("lena.png not found")

# A. Manual brightness increase
bright = np.clip(image.astype(np.int16) + 50, 0, 255).astype(np.uint8)

# B. Manual negative transformation
negative = 255 - image

# C. Manual binary thresholding
threshold = np.where(image > 127, 255, 0).astype(np.uint8)

# Display results
titles = [
    "Original Image",
    "Manual Brightness",
    "Manual Negative",
    "Manual Thresholding"
]

images = [image, bright, negative, threshold]

plt.figure(figsize=(10, 6))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap="gray", vmin=0, vmax=255)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
