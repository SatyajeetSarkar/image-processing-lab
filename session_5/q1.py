import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Add Gaussian noise
noise = np.random.normal(0, 25, image.shape)
noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# Apply image averaging using multiple noisy images
noisy_images = []
for i in range(5):
    noise = np.random.normal(0, 25, image.shape)
    temp = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    noisy_images.append(temp.astype(np.float32))

averaged = np.mean(noisy_images, axis=0).astype(np.uint8)

# Apply filters with different kernel sizes
results = [image, noisy, averaged]
titles = ["Original", "Noisy Image", "Image Averaging"]

for k in [3, 5, 7]:
    # Apply mean filter
    mean = cv2.blur(noisy, (k, k))

    # Apply Gaussian filter
    gaussian = cv2.GaussianBlur(noisy, (k, k), 0)

    # Apply low-pass filter using averaging kernel
    kernel = np.ones((k, k), np.float32) / (k * k)
    low_pass = cv2.filter2D(noisy, -1, kernel)

    results.extend([mean, gaussian, low_pass])
    titles.extend([f"Mean {k}x{k}", f"Gaussian {k}x{k}", f"Low-pass {k}x{k}"])

# Display all results
plt.figure(figsize=(14, 12))

for i in range(len(results)):
    plt.subplot(4, 3, i + 1)
    plt.imshow(results[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
