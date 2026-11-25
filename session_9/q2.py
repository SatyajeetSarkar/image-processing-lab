import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Perform one-level DWT
level1 = pywt.wavedec2(image, "haar", level=1)

# Perform two-level DWT
level2 = pywt.wavedec2(image, "haar", level=2)

# Perform three-level DWT
level3 = pywt.wavedec2(image, "haar", level=3)

# Reconstruct images using inverse DWT
reconstructed1 = pywt.waverec2(level1, "haar")
reconstructed2 = pywt.waverec2(level2, "haar")
reconstructed3 = pywt.waverec2(level3, "haar")

# Convert reconstructed images to uint8
reconstructed1 = np.clip(reconstructed1, 0, 255).astype(np.uint8)
reconstructed2 = np.clip(reconstructed2, 0, 255).astype(np.uint8)
reconstructed3 = np.clip(reconstructed3, 0, 255).astype(np.uint8)

# Calculate reconstruction error
print("Level 1 MSE:", np.mean((image.astype(float) - reconstructed1.astype(float))**2))
print("Level 2 MSE:", np.mean((image.astype(float) - reconstructed2.astype(float))**2))
print("Level 3 MSE:", np.mean((image.astype(float) - reconstructed3.astype(float))**2))

# Display original and reconstructed images
images = [image, reconstructed1, reconstructed2, reconstructed3]
titles = ["Original", "1-Level Reconstruction", "2-Level Reconstruction", "3-Level Reconstruction"]

plt.figure(figsize=(12, 7))

for i in range(len(images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
