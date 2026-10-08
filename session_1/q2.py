import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

# Check if image was loaded successfully
if image is None:
    print("Error: Could not read the image.")
    exit()


brightness_increased = cv2.add(image, 50)

brightness_decreased = cv2.subtract(image, 50)


contrast_increased = cv2.convertScaleAbs(
    image,
    alpha=1.5,
    beta=0
)

contrast_decreased = cv2.convertScaleAbs(
    image,
    alpha=0.5,
    beta=0
)


negative = 255 - image

threshold_value = 128

_, binary = cv2.threshold(
    image,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)


images = [
    image,
    brightness_increased,
    brightness_decreased,
    contrast_increased,
    contrast_decreased,
    negative,
    binary
]

titles = [
    "Original Image",
    "Increased Brightness",
    "Decreased Brightness",
    "Increased Contrast",
    "Decreased Contrast",
    "Negative Image",
    "Binary Threshold"
]

plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


manual_brightness = image.astype(np.int16) + 50

# Keep values within 0-255
manual_brightness = np.clip(manual_brightness, 0, 255)

# Convert back to uint8
manual_brightness = manual_brightness.astype(np.uint8)


manual_negative = 255 - image


manual_binary = np.where(image >= 128, 255, 0)

manual_binary = manual_binary.astype(np.uint8)


manual_images = [
    manual_brightness,
    manual_negative,
    manual_binary
]

manual_titles = [
    "Manual Brightness - NumPy",
    "Manual Negative - NumPy",
    "Manual Binary Threshold - NumPy"
]

plt.figure(figsize=(12, 4))

for i in range(len(manual_images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(manual_images[i], cmap="gray")
    plt.title(manual_titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()