import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Convert image to float and normalize
image = image.astype(np.float32) / 255.0

# Apply logarithmic transformation
log_image = np.log1p(image)

# Calculate and shift the Fourier transform
dft = np.fft.fft2(log_image)
dft_shift = np.fft.fftshift(dft)

# Get image dimensions and center
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Create distance matrix
y, x = np.ogrid[:rows, :cols]
distance = (y - crow)**2 + (x - ccol)**2

# Set homomorphic filter parameters
gamma_l = 0.5
gamma_h = 1.5
D0 = 30
c = 1

# Construct homomorphic filter
H = (gamma_h - gamma_l) * (1 - np.exp(-c * distance / (D0**2))) + gamma_l

# Apply the filter in frequency domain
filtered = dft_shift * H

# Perform inverse Fourier transform
inverse = np.fft.ifftshift(filtered)
inverse_dft = np.fft.ifft2(inverse)

# Apply exponential transformation
enhanced = np.expm1(np.real(inverse_dft))

# Normalize enhanced image
enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
enhanced = enhanced.astype(np.uint8)

# Display original and enhanced images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow((image * 255).astype(np.uint8), cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(enhanced, cmap="gray")
plt.title("Homomorphic Enhanced")
plt.axis("off")

plt.tight_layout()
plt.show()
