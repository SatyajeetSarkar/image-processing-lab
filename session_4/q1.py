import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Create four image variations
dark = cv2.subtract(image, 100)
bright = cv2.add(image, 100)
low_contrast = cv2.convertScaleAbs(image, alpha=0.5, beta=80)
high_contrast = cv2.convertScaleAbs(image, alpha=2.0, beta=-100)

images = {
    "Dark Image": dark,
    "Bright Image": bright,
    "Low Contrast Image": low_contrast,
    "High Contrast Image": high_contrast
}

plt.figure(figsize=(12, 12))

for i, (name, img) in enumerate(images.items()):
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    cdf = np.cumsum(histogram)
    cdf = cdf / cdf[-1]

    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(img, cmap="gray")
    plt.title(name)
    plt.axis("off")

    plt.subplot(4, 3, i * 3 + 2)
    plt.plot(histogram)
    plt.title("Histogram")
    plt.xlim([0, 255])

    plt.subplot(4, 3, i * 3 + 3)
    plt.plot(cdf)
    plt.title("CDF")
    plt.xlim([0, 255])
    plt.ylim([0, 1])

plt.tight_layout()
plt.show()
