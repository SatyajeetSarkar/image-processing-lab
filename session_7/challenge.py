import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Read the grayscale image
image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully")

# Calculate and shift the Fourier transform
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Get image dimensions and center
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Create distance matrix
y, x = np.ogrid[:rows, :cols]
distance = np.sqrt((y - crow)**2 + (x - ccol)**2)

# Apply Gaussian low-pass filter
def filter_image(cutoff):
    mask = np.exp(-(distance**2) / (2 * cutoff**2))
    filtered = dft_shift * mask
    inverse = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.abs(inverse)

# Create initial filtered image
initial_cutoff = 30
result = filter_image(initial_cutoff)

# Display original and filtered images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

display_image = ax[1].imshow(result, cmap="gray")
ax[1].set_title("Gaussian Low-pass Filter")
ax[1].axis("off")

# Create cut-off frequency slider
slider_ax = plt.axes([0.2, 0.08, 0.6, 0.04])
cutoff_slider = Slider(slider_ax, "Cut-off Frequency", 5, 100, valinit=30)

# Update image when slider changes
def update(value):
    cutoff = cutoff_slider.val
    filtered = filter_image(cutoff)
    display_image.set_data(filtered)
    fig.canvas.draw_idle()

cutoff_slider.on_changed(update)

plt.show()
