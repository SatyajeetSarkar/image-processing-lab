import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Add salt-and-pepper noise
noisy = image.copy()
probability = 0.05
random = np.random.random(image.shape)

noisy[random < probability / 2] = 0
noisy[random > 1 - probability / 2] = 255

# Store original and noisy images
results = [image, noisy]
titles = ["Original", "Salt-and-Pepper Noise"]

for k in [3, 5, 7]:
    # Apply median filter
    median = cv2.medianBlur(noisy, k)

    # Apply minimum filter using erosion
    minimum = cv2.erode(noisy, np.ones((k, k), np.uint8))

    # Apply maximum filter using dilation
    maximum = cv2.dilate(noisy, np.ones((k, k), np.uint8))

    results.extend([median, minimum, maximum])
    titles.extend([f"Median {k}x{k}", f"Minimum {k}x{k}", f"Maximum {k}x{k}"])

# Display all results
plt.figure(figsize=(14, 12))

for i in range(len(results)):
    plt.subplot(4, 3, i + 1)
    plt.imshow(results[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
