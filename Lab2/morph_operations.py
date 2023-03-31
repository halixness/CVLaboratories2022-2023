import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('..\Lab1\material\j.png')
assert img is not None, "Could not read image"

# Show it
# cv2.imshow('image', img)
# cv2.waitKey(30000)

# Erosion operation 
kernel = np.ones((5,5), dtype = np.uint8)
erosion = cv2.erode(img, kernel, iterations = 1)

# 1 row, 2 columns, 1 index
# plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title("Original")
# plt.subplot(122), plt.imshow(erosion, cmap="gray"), plt.title("Erosion")
# plt.show()

# Dilation operation 
kernel = np.ones((5,5), dtype = np.uint8)
erosion = cv2.dilate(img, kernel, iterations = 1)

# plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title("Original")
# plt.subplot(122), plt.imshow(erosion, cmap="gray"), plt.title("Dilation")
# plt.show()

# Morphology Opening: erosion + dilation
img = cv2.imread("..\Lab1\material\opening.png", cv2.IMREAD_GRAYSCALE)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title("Original")
# plt.subplot(122), plt.imshow(opening, cmap="gray"), plt.title("Opening")
# plt.show()

# Morphology Closing: dilation + erosion
img = cv2.imread("..\Lab1\material\closing.png", cv2.IMREAD_GRAYSCALE)
opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(122), plt.imshow(opening, cmap="gray"), plt.title("Closing")

plt.show()

