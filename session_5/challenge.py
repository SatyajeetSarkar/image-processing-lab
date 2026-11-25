import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")
    
# Add salt-and-pepper noise
noisy = image.copy()
random = np.random.random(image.shape)
noisy[random < 0.025] = 0
noisy[random > 0.975] = 255

# Apply OpenCV median filter
opencv_result = cv2.medianBlur(noisy, 3)

# Add padding around the noisy image
padded = np.pad(noisy, 1, mode="edge")

# Create an empty output image
manual_result = np.zeros_like(noisy)

# Apply the median operation manually
for i in range(noisy.shape[0]):
    for j in range(noisy.shape[1]):
        window = padded[i:i+3, j:j+3]
        manual_result[i, j] = np.median(window)

# Compare both results
difference = cv2.absdiff(manual_result, opencv_result)
print("Maximum pixel difference:", np.max(difference))
print("Mean pixel difference:", np.mean(difference))

# Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(manual_result, cmap="gray")
plt.title("Manual Median")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(opencv_result, cmap="gray")
plt.title("OpenCV Median")
plt.axis("off")

plt.tight_layout()
plt.show()
