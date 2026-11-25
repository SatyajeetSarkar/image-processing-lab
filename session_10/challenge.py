import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("unknown.png")

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reduce noise before circle detection
blur = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles
circles = cv2.HoughCircles(
    blur, cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=100,
    param2=35,
    minRadius=10,
    maxRadius=150
)

result = image.copy()
count = 0

# Draw and count detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))

    for x, y, r in circles[0]:
        count += 1
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(result, str(count), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)

print("Total circles detected:", count)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Detected Circles")
plt.axis("off")

plt.tight_layout()
plt.show()
