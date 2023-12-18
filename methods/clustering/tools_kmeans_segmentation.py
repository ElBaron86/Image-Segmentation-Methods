'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-12-18 19:23:18
 # @ Modified by:
 # @ Modified time: 
 # @ Description: Function to apply k-means segmentation on a colored image.
 '''

# Modules importations
import cv2
import numpy as np
from sklearn.cluster import KMeans

def kmeans_segmentation(image_path: str, k: int = 7) -> np.ndarray:
    """
    Perform image segmentation using k-means clustering.

    Args:
    - image_path (str): Path to the input image.
    - k (int): Number of clusters for k-means clustering.

    Returns:
    - np.ndarray: Segmented image.
    """
    # Load the image
    image = cv2.imread(image_path)  # noqa
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # noqa

    # Preprocessing: reshape the image into a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Converting to dtype float32 before using K-means
    pixels = np.float32(pixels)

    # Apply the k-means algorithm
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get cluster labels for each pixel
    labels = kmeans.labels_

    # Assign each pixel to the color of the corresponding centroid
    segmented_image = kmeans.cluster_centers_.astype(int)[labels]

    # Reshape the image back to its original form
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image
