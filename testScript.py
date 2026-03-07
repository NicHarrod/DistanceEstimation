import cv2
import numpy as np
import argparse
import os

# Import your class
from depth_anything_v3_pytorch import Depth_Anything_3  
from depth_anything import DepthAnything
import cv2

model = Depth_Anything_3()

img = cv2.imread("../small_example/adk01/transects/transect1/detection_frames/8230044_51.jpg")

depth = model(img)

model2 = DepthAnything()

depth2 = model2(img)

print(depth.shape)
print(depth2.shape)
import matplotlib.pyplot as plt

# Normalize both depth maps
depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
depth2_norm = (depth2 - depth2.min()) / (depth2.max() - depth2.min())

# Convert BGR to RGB for correct display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Depth Anything V3")
plt.imshow(depth_norm, cmap="inferno")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Depth Anything (old)")
plt.imshow(depth2_norm, cmap="inferno")
plt.axis("off")

plt.tight_layout()
plt.show()