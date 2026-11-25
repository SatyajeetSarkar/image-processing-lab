import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

height, width = image.shape

# A. Translation
translation_matrix = np.float32([
    [1, 0, 50],
    [0, 1, 30]
])

translated = cv2.warpAffine(
    image, translation_matrix, (width, height)
)

# B. Scaling
scaled = cv2.resize(
    image, None, fx=1.5, fy=1.5,
    interpolation=cv2.INTER_LINEAR
)

# C. Rotation
center = (width // 2, height // 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center, 45, 1
)

rotated = cv2.warpAffine(
    image, rotation_matrix, (width, height)
)

# D. Horizontal reflection
horizontal = cv2.flip(image, 1)

# E. Vertical reflection
vertical = cv2.flip(image, 0)

# Display results
titles = [
    "Original Image",
    "Translation",
    "Scaling",
    "Rotation",
    "Horizontal Reflection",
    "Vertical Reflection"
]

images = [
    image, translated, scaled,
    rotated, horizontal, vertical
]

plt.figure(figsize=(12, 8))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
