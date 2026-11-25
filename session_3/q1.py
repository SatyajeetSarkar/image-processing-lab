import numpy as np
import matplotlib.pyplot as plt

# Create a sample binary image
binary = np.array([
    [0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1]
], dtype=np.uint8)

# Select a pixel
x, y = 2, 2
value = binary[y, x]

# Define 4-neighbour
four_directions = [
    (-1, 0), (1, 0), (0, -1), (0, 1)
]

# Diagonal positions
diagonal_directions = [
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]

height, width = binary.shape

# Find 4-neighbours
four_neighbors = []

for dy, dx in four_directions:
    ny, nx = y + dy, x + dx

    if 0 <= ny < height and 0 <= nx < width:
        if binary[ny, nx] == 1:
            four_neighbors.append((nx, ny))

# Find 8-neighbours
eight_neighbors = []

for dy, dx in four_directions + diagonal_directions:
    ny, nx = y + dy, x + dx

    if 0 <= ny < height and 0 <= nx < width:
        if binary[ny, nx] == 1:
            eight_neighbors.append((nx, ny))

# Find m-neighbours
m_neighbors = []

for dy, dx in four_directions:
    ny, nx = y + dy, x + dx

    if 0 <= ny < height and 0 <= nx < width:
        if binary[ny, nx] == 1:
            m_neighbors.append((nx, ny))

for dy, dx in diagonal_directions:
    ny, nx = y + dy, x + dx

    if 0 <= ny < height and 0 <= nx < width:
        if binary[ny, nx] == 1:
            common = [
                (y, nx),
                (ny, x)
            ]

            # Diagonal neighbours connect only if common 4-neighbours are zero
            if all(binary[cy, cx] == 0 for cy, cx in common):
                m_neighbors.append((nx, ny))

# Display connectivity results
print("Selected pixel (x, y):", (x, y))
print("Pixel value:", value)
print("4-neighbours:", four_neighbors)
print("8-neighbours:", eight_neighbors)
print("m-neighbours:", m_neighbors)

# Calculate distances between two pixels
p1 = (2, 2)
p2 = (5, 6)

dx = abs(p2[0] - p1[0])
dy = abs(p2[1] - p1[1])

euclidean = np.sqrt(dx**2 + dy**2)
city_block = dx + dy
chessboard = max(dx, dy)

print("\nDistance between", p1, "and", p2)
print("Euclidean Distance:", euclidean)
print("City-block Distance:", city_block)
print("Chessboard Distance:", chessboard)

# Display binary image
plt.imshow(binary, cmap="gray", vmin=0, vmax=1)
plt.title("Sample Binary Image")
plt.colorbar()
plt.show()
