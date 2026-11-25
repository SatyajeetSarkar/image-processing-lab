import cv2
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Define interpolation techniques
methods = {
    "Nearest Neighbor": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC
}

# Define scaling factors
scales = [2, 4, 0.5, 0.25]

# Display original image
plt.figure(figsize=(15, 12))

plt.subplot(4, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plot_index = 2

# Perform zooming and shrinking
for scale in scales:
    for method_name, method in methods.items():

        resized = cv2.resize(
            image, None,
            fx=scale, fy=scale,
            interpolation=method
        )

        plt.subplot(4, 4, plot_index)
        plt.imshow(resized, cmap="gray")
        plt.title(f"{method_name}\nScale: {scale}x")
        plt.axis("off")

        plot_index += 1

plt.tight_layout()
plt.show()
