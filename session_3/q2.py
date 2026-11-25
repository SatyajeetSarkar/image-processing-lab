import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# Read grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# A. Calculate 2-D DFT
dft = np.fft.fft2(image)

# B. Shift zero-frequency component to centre
dft_shift = np.fft.fftshift(dft)

# C. Calculate magnitude spectrum
magnitude = np.abs(dft_shift)

# D. Apply logarithmic scaling
log_magnitude = 20 * np.log(magnitude + 1)

# E. Calculate 2-D DCT
dct_result = dct(
    dct(image.astype(np.float32), axis=0, norm="ortho"),
    axis=1, norm="ortho"
)

# F. Display DCT coefficient matrix
print("DCT Coefficient Matrix (First 10x10):")
print(np.round(dct_result[:10, :10], 2))

# Display results
titles = [
    "Original Image",
    "DFT Magnitude Spectrum",
    "Log Magnitude Spectrum",
    "DCT Coefficients"
]

images = [
    image,
    magnitude,
    log_magnitude,
    np.log(np.abs(dct_result) + 1)
]

plt.figure(figsize=(12, 8))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
