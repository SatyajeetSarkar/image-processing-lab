import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Convert image to float
original = image.astype(np.float32) / 255.0

# Create a Gaussian blur kernel
kernel = cv2.getGaussianKernel(9, 2)
psf = kernel @ kernel.T

# Apply blur to create degradation
blurred = convolve2d(original, psf, mode="same", boundary="symm")

# Add Gaussian noise
noise = np.random.normal(0, 0.01, blurred.shape)
degraded = np.clip(blurred + noise, 0, 1)

# Calculate Fourier transforms
H = np.fft.fft2(psf, s=original.shape)
G = np.fft.fft2(degraded)

# Inverse filtering with a small stabilizing constant
inverse = G * np.conj(H) / (np.abs(H)**2 + 1e-3)
inverse_result = np.abs(np.fft.ifft2(inverse))
inverse_result = np.clip(inverse_result, 0, 1)

# Wiener filtering
K = 0.001
wiener = G * np.conj(H) / (np.abs(H)**2 + K)
wiener_result = np.abs(np.fft.ifft2(wiener))
wiener_result = np.clip(wiener_result, 0, 1)

# Least-square regularized restoration
laplacian = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]], dtype=np.float32)

P = np.fft.fft2(laplacian, s=original.shape)
gamma = 0.01

least_square = G * np.conj(H) / (np.abs(H)**2 + gamma * np.abs(P)**2)
least_square_result = np.abs(np.fft.ifft2(least_square))
least_square_result = np.clip(least_square_result, 0, 1)

# Calculate MSE
def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

# Calculate PSNR
def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)

# Compare restoration quality
results = {
    "Degraded": degraded,
    "Inverse": inverse_result,
    "Wiener": wiener_result,
    "Least-Square": least_square_result
}

print("Restoration Quality")
for name, result in results.items():
    print(name, "MSE:", calculate_mse(original, result))
    print(name, "PSNR:", calculate_psnr(original, result), "dB")

# Display results
images = [
    original,
    degraded,
    inverse_result,
    wiener_result,
    least_square_result
]

titles = [
    "Original",
    "Degraded",
    "Inverse Filtering",
    "Wiener Filtering",
    "Least-Square Restoration"
]

plt.figure(figsize=(14, 8))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray", vmin=0, vmax=1)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
