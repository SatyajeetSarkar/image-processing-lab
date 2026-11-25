import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Linear contrast stretching
min_val = np.min(image)
max_val = np.max(image)

linear = ((image.astype(np.float32) - min_val) * 255 /
          (max_val - min_val)).astype(np.uint8)

# Non-linear contrast stretching using gamma correction
gamma = 0.5
normalized = image.astype(np.float32) / 255.0
nonlinear = np.clip((normalized ** gamma) * 255, 0, 255).astype(np.uint8)

# Histogram equalization
equalized = cv2.equalizeHist(image)

results = {
    "Linear Contrast Stretching": linear,
    "Non-linear (Gamma) Correction": nonlinear,
    "Histogram Equalization": equalized
}

plt.figure(figsize=(12, 12))

for i, (name, enhanced) in enumerate(results.items()):
    original_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    enhanced_hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256])

    plt.subplot(3, 4, i * 4 + 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(3, 4, i * 4 + 2)
    plt.imshow(enhanced, cmap="gray")
    plt.title(name)
    plt.axis("off")

    plt.subplot(3, 4, i * 4 + 3)
    plt.plot(original_hist)
    plt.title("Original Histogram")
    plt.xlim([0, 255])

    plt.subplot(3, 4, i * 4 + 4)
    plt.plot(enhanced_hist)
    plt.title("Enhanced Histogram")
    plt.xlim([0, 255])

plt.tight_layout()
plt.show()
