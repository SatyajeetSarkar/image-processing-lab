import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the color image
image = cv2.imread("lena.png")

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Convert BGR to RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Separate RGB channels
red, green, blue = cv2.split(rgb)

# Modify individual color channels
red_modified = np.clip(red.astype(np.int16) + 50, 0, 255).astype(np.uint8)
green_modified = np.clip(green.astype(np.int16) + 50, 0, 255).astype(np.uint8)
blue_modified = np.clip(blue.astype(np.int16) + 50, 0, 255).astype(np.uint8)

# Reconstruct modified color image
modified = cv2.merge([red_modified, green_modified, blue_modified])

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Display all results
images = [
    rgb,
    red,
    green,
    blue,
    modified,
    gray,
    hsv
]

titles = [
    "Original RGB",
    "Red Channel",
    "Green Channel",
    "Blue Channel",
    "Modified RGB",
    "Grayscale",
    "HSV Image"
]

plt.figure(figsize=(14, 10))

for i in range(len(images)):
    plt.subplot(3, 3, i + 1)

    if titles[i] == "HSV Image":
        plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    elif titles[i] == "Grayscale" or i in [1, 2, 3]:
        plt.imshow(images[i], cmap="gray")
    else:
        plt.imshow(images[i])

    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
