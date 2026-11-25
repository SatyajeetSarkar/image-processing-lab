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
LL, (LH, HL, HH) = pywt.dwt2(image, "haar")

# Remove all high-frequency coefficients
LH_removed = np.zeros_like(LH)
HL_removed = np.zeros_like(HL)
HH_removed = np.zeros_like(HH)

# Reconstruct image using only LL component
reconstructed = pywt.idwt2((LL, (LH_removed, HL_removed, HH_removed)), "haar")

# Convert reconstructed image to uint8
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

# Display original and reconstructed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap="gray")
plt.title("High-Frequency Removed")
plt.axis("off")

plt.tight_layout()
plt.show()
