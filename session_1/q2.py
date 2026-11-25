import cv2
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread("lena.png", 0)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# A. Increase brightness
bright = cv2.add(image, 50)

# B. Decrease brightness
dark = cv2.subtract(image, 50)

# C. Increase contrast
high_contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

# D. Decrease contrast
low_contrast = cv2.convertScaleAbs(image, alpha=0.5, beta=0)

# E. Generate negative image
negative = 255 - image

# F. Binary thresholding
_, threshold = cv2.threshold(
    image, 127, 255, cv2.THRESH_BINARY
)

# Display all images
titles = [
    "Original Image",
    "Increased Brightness",
    "Decreased Brightness",
    "Increased Contrast",
    "Decreased Contrast",
    "Negative Image",
    "Binary Thresholding"
]

images = [
    image, bright, dark, high_contrast,
    low_contrast, negative, threshold
]

plt.figure(figsize=(12, 9))

for i in range(7):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap="gray", vmin=0, vmax=255)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
