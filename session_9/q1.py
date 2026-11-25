import cv2
import pywt
import matplotlib.pyplot as plt

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Perform one-level 2D DWT
LL, (LH, HL, HH) = pywt.dwt2(image, "haar")

# Display all components
images = [image, LL, LH, HL, HH]
titles = ["Original Image", "LL Component", "LH Component", "HL Component", "HH Component"]

plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
