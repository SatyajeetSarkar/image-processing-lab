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

# Define Sobel kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

# Define manual Sobel function
def manual_sobel(img):
    padded = np.pad(img.astype(np.float32), 1, mode="edge")
    gx = np.zeros(img.shape, dtype=np.float32)
    gy = np.zeros(img.shape, dtype=np.float32)

    # Perform convolution at every pixel
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    # Calculate gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)
    return cv2.convertScaleAbs(magnitude)

# Apply Sobel to clean and noisy images
clean_edges = manual_sobel(image)
noisy_edges = manual_sobel(noisy)

# Display results
images = [image, clean_edges, noisy, noisy_edges]
titles = ["Clean Image", "Clean Sobel Edges", "Noisy Image", "Noisy Sobel Edges"]

plt.figure(figsize=(12, 7))

for i in range(len(images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
