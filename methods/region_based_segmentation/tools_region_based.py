'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-12-17 16:22:46
 # @ Description: Functions to apply a segmentation by region on an image (color).
 '''

# Modules importation
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Region growing function
def region_growing(img: np.ndarray, seed: Tuple[int, int] = (50, 50), texture_threshold : int = 64) -> np.ndarray:
    """
    Applies region growing segmentation to an image.

    Parameters:
        img (np.ndarray): The image to be segmented.
        seed (Tuple[int, int]): The coordinates of the initial seed. Defautl (50, 50)
        texture_threshold (int): Texture threshold parameters

    Returns:
        np.ndarray: The segmented image.

    Raises:
        ValueError: If the dimensions of the image are not valid.
    """

    # Check image dimensions
    if len(img.shape) != 3:
        raise ValueError("The image must be in color (3 channels).")

    # Create an output image initialized with zeros
    h, w = img.shape[:2]
    segmented = np.zeros((h, w), dtype=np.uint8)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the stack for region growing
    stack = [seed]
    while stack:
        x, y = stack.pop()

        # Check if the pixel has not been visited yet
        if segmented[x, y] == 0:
            # Check if the texture of the pixel is similar to the seed
            if np.abs(int(gray_img[x, y]) - int(gray_img[seed])) < texture_threshold:
                # Add the pixel to the region
                segmented[x, y] = 255

                # Add neighbors to the stack
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= x + i < h and 0 <= y + j < w:
                            stack.append((x + i, y + j))

    return segmented
