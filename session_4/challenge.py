import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Calculate histogram and CDF
histogram = np.bincount(image.ravel(), minlength=256)
cdf = np.cumsum(histogram)

# Normalize the CDF
cdf_min = cdf[np.nonzero(cdf)[0][0]]
total_pixels = image.size

# Create lookup table
if total_pixels > cdf_min:
    lookup_table = np.round(
        (cdf - cdf_min) * 255 / (total_pixels - cdf_min)
    )
    lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)
else:
    lookup_table = np.arange(256, dtype=np.uint8)

# Apply manual histogram equalization
manual_equalized = lookup_table[image]

# OpenCV histogram equalization
opencv_equalized = cv2.equalizeHist(image)

# Compare the results
difference = cv2.absdiff(manual_equalized, opencv_equalized)

print("Maximum pixel difference:",
      np.max(difference))

print("Mean pixel difference:",
      np.mean(difference))

# Display results
plt.figure(figsize=(12, 8))

images = [
    (image, "Original Image"),
    (manual_equalized, "Manual Equalization"),
    (opencv_equalized, "OpenCV Equalization"),
    (difference, "Difference")
]

for i, (img, title) in enumerate(images):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
