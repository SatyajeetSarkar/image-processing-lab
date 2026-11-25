import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Calculate the 2D DFT and shift the zero frequency to the center
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Get image dimensions and center coordinates
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Set cutoff frequency
D0 = 30

# Create distance matrix from the frequency center
y, x = np.ogrid[:rows, :cols]
distance = np.sqrt((y - crow)**2 + (x - ccol)**2)

# Generate Ideal Low-pass Filter
ideal_low = np.zeros((rows, cols), np.float32)
ideal_low[distance <= D0] = 1

# Generate Gaussian Low-pass Filter
gaussian_low = np.exp(-(distance**2) / (2 * D0**2))

# Generate Ideal High-pass Filter
ideal_high = 1 - ideal_low

# Generate Gaussian High-pass Filter
gaussian_high = 1 - gaussian_low

# Apply filter and perform inverse DFT
def apply_filter(mask):
    filtered = dft_shift * mask
    inverse = np.fft.ifftshift(filtered)
    result = np.fft.ifft2(inverse)
    return np.abs(result)

# Apply all four filters
results = [
    image,
    apply_filter(ideal_low),
    apply_filter(gaussian_low),
    apply_filter(ideal_high),
    apply_filter(gaussian_high)
]

titles = [
    "Original",
    "Ideal Low-pass",
    "Gaussian Low-pass",
    "Ideal High-pass",
    "Gaussian High-pass"
]

# Display the results
plt.figure(figsize=(12, 8))

for i in range(len(results)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(results[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
