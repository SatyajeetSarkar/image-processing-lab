import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

height, width = image.shape

# Pad image dimensions to multiples of 8
padded = cv2.copyMakeBorder(
    image, 0, (8 - height % 8) % 8,
    0, (8 - width % 8) % 8,
    cv2.BORDER_REFLECT
)

# Standard JPEG-like quantization matrix
quant_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

padded = padded.astype(np.float32) - 128

reconstructed = np.zeros_like(padded)
quantized_blocks = []

# Divide image into 8x8 blocks and apply DCT
for y in range(0, padded.shape[0], 8):
    for x in range(0, padded.shape[1], 8):
        block = padded[y:y+8, x:x+8]

        dct_block = cv2.dct(block)

        # Quantize DCT coefficients
        quantized = np.round(dct_block / quant_matrix).astype(np.int16)
        quantized_blocks.append(quantized)

        # Reconstruct the block using inverse DCT
        restored_dct = quantized.astype(np.float32) * quant_matrix
        restored_block = cv2.idct(restored_dct)

        reconstructed[y:y+8, x:x+8] = restored_block

# Restore original image dimensions
reconstructed = np.clip(reconstructed[:height, :width] + 128, 0, 255)
reconstructed = reconstructed.astype(np.uint8)

# Calculate MSE and PSNR
mse = np.mean((image.astype(np.float32) -
               reconstructed.astype(np.float32)) ** 2)

psnr = float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)

# Save reconstructed image
cv2.imwrite("dct_reconstructed.png", reconstructed)

# Save quantized coefficients as compressed data
np.savez_compressed("dct_coefficients.npz",
                    coefficients=np.array(quantized_blocks, dtype=np.int16))

# Calculate actual file-size compression ratio
original_size = os.path.getsize("lena.png")
compressed_size = os.path.getsize("dct_coefficients.npz")
compression_ratio = original_size / compressed_size

print("Original file size:", original_size, "bytes")
print("Compressed coefficient file size:", compressed_size, "bytes")
print("Compression ratio:", round(compression_ratio, 2))
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
plt.title("DCT Reconstructed Image")
plt.axis("off")

plt.tight_layout()
plt.show()
