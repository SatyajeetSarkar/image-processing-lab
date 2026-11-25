import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Read original image
image = cv2.imread("lena.png")

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

original_size = os.path.getsize("lena.png")

# Calculate MSE
def calculate_mse(original, compressed):
    return np.mean((original.astype(np.float32) -
                    compressed.astype(np.float32)) ** 2)

# Calculate PSNR
def calculate_psnr(mse):
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255 ** 2) / mse)

# Save PNG and JPEG versions
cv2.imwrite("compressed.png", image)

for quality in [90, 70, 50, 20]:
    cv2.imwrite(
        f"compressed_q{quality}.jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

# Compare compression results
files = ["compressed.png",
         "compressed_q90.jpg",
         "compressed_q70.jpg",
         "compressed_q50.jpg",
         "compressed_q20.jpg"]

print("Original file size:", original_size, "bytes")
print("\nCompression Results")
print("-" * 75)
print(f"{'Format':<20}{'Size (bytes)':<15}{'Ratio':<12}{'MSE':<12}{'PSNR (dB)'}")

compressed_images = []

for filename in files:
    compressed = cv2.imread(filename)

    if compressed is None:
        print("Could not read:", filename)
        continue

    file_size = os.path.getsize(filename)
    ratio = original_size / file_size
    mse = calculate_mse(image, compressed)
    psnr = calculate_psnr(mse)

    print(f"{filename:<20}{file_size:<15}{ratio:<12.2f}{mse:<12.2f}{psnr:.2f}")

    compressed_images.append((filename, compressed))

# Display compressed images
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

for i, (filename, compressed) in enumerate(compressed_images):
    plt.subplot(2, 3, i + 2)
    plt.imshow(cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB))
    plt.title(filename)
    plt.axis("off")

plt.tight_layout()
plt.show()
