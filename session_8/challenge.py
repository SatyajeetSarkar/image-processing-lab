import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the color image
image = cv2.imread("lena_color.png")

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Convert BGR image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define green color range
lower = np.array([35, 40, 40])
upper = np.array([85, 255, 255])

# Create a color detection mask
mask = cv2.inRange(hsv, lower, upper)

# Extract the selected color
extracted = cv2.bitwise_and(image, image, mask=mask)

# Convert images to RGB for display
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)

# Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Green Color Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(extracted_rgb)
plt.title("Extracted Green")
plt.axis("off")

plt.tight_layout()
plt.show()
