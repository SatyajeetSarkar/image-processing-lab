import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# Read grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Convert image to floating point
image_float = image.astype(np.float32)

# Calculate 2-D DCT
dct_result = dct(
    dct(image_float, axis=0, norm="ortho"),
    axis=1, norm="ortho"
)

# Select percentage of low-frequency coefficients
percentage = 25

height, width = image.shape

# Calculate retained coefficient dimensions
keep_h = max(1, int(height * np.sqrt(percentage / 100)))
keep_w = max(1, int(width * np.sqrt(percentage / 100)))

# Create matrix containing only low-frequency coefficients
filtered_dct = np.zeros_like(dct_result)

filtered_dct[:keep_h, :keep_w] = dct_result[:keep_h, :keep_w]

# Apply inverse DCT
reconstructed = idct(
    idct(filtered_dct, axis=0, norm="ortho"),
    axis=1, norm="ortho"
)

# Convert reconstructed image to uint8
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

# Calculate MSE
mse = np.mean(
    (image.astype(np.float32) -
     reconstructed.astype(np.float32)) ** 2
)

# Calculate PSNR
psnr = float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)

# Estimate coefficient retention ratio
total_coefficients = height * width
retained_coefficients = keep_h * keep_w

compression_ratio = total_coefficients / retained_coefficients

print("Selected Percentage:", percentage, "%")
print("Retained Coefficients:", retained_coefficients)
print("Total Coefficients:", total_coefficients)
print("Coefficient Compression Ratio:", round(compression_ratio, 2))
print("MSE:", round(mse, 2))
print("PSNR:", round(psnr, 2), "dB")

# Display original and reconstructed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap="gray")
plt.title(f"Reconstructed ({percentage}% coefficients)")
plt.axis("off")

plt.tight_layout()
plt.show()
