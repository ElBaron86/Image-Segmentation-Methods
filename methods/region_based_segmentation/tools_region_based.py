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
        gray_img = img
    else:
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Create an output image initialized with zeros
    h, w = img.shape[:2]
    segmented = np.zeros((h, w), dtype=np.uint8)

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

# I'm working on it...

# Split/merge function
def split_and_merge(image: np.ndarray, min_region_size: int) -> np.ndarray:
    """
    Perform split-and-merge segmentation on the input image.

    Args:
    - image (np.ndarray): The input image.
    - min_region_size (int): The minimum size of a region for splitting.

    Returns:
    - np.ndarray: The segmented image.
    """
    # Check if the image is already in grayscale
    if len(image.shape) == 2:
        gray_img = image
    else:
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def split(image: np.ndarray) -> list[np.ndarray]:
        """
        Split the image into four quadrants.

        Args:
        - image (np.ndarray): The input image.

        Returns:
        - list[np.ndarray]: List of four quadrants.
        """
        quadrants = np.array_split(image, 2, axis=0)
        quadrants = [np.array_split(quadrant, 2, axis=1) for quadrant in quadrants]
        return [subquadrant for quadrant in quadrants for subquadrant in quadrant]

    def merge(regions: list[np.ndarray]) -> np.ndarray:
        """
        Merge list of regions by averaging their colors.

        Args:
        - regions (list[np.ndarray]): List of regions to be merged.

        Returns:
        - np.ndarray: Merged region.
        """
        average_color = np.average(regions, axis=0)
        return np.full_like(regions[0], average_color, dtype=np.uint8)

    queue = [gray_img]

    while queue:
        current_region = queue.pop(0)

        if current_region.size <= min_region_size:
            continue

        subregions = split(current_region)

        if all(np.all(subregion == subregions[0]) for subregion in subregions):
            queue.append(merge(subregions))
        else:
            queue.extend(subregions)

    return merge(subregions)