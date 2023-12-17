import os
import sys
import numpy as np

# Moving up to the `Image-Segmentation-Methods` directory
while os.path.basename(os.getcwd()) != "Image-Segmentation-Methods":
    os.chdir("..")
sys.path.append(os.getcwd())


from methods.region_based_segmentation.tools_region_based import region_growing

import unittest

class TestRegionGrowing(unittest.TestCase):

    # Applies region growing segmentation to an image
    def test_applies_region_growing(self):
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = [255, 255, 255]

        # Apply region growing segmentation
        segmented = region_growing(img)

        # Check if the segmented image is not empty
        self.assertTrue(np.any(segmented != 0))

    # Returns the segmented image
    def test_returns_segmented_image(self):
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = [255, 255, 255]

        # Apply region growing segmentation
        segmented = region_growing(img)

        # Check if the segmented image has the same shape as the original image
        self.assertEqual(segmented.shape, img.shape[:2])

    # Uses default seed if not provided
    def test_uses_default_seed(self):
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = [255, 255, 255]

        # Apply region growing segmentation without specifying seed
        segmented = region_growing(img)

        # Check if the segmented image is not empty
        self.assertTrue(np.any(segmented != 0))

    # Raises ValueError if image dimensions are not valid
    def test_raises_value_error(self):
        # Create a grayscale test image
        img = np.zeros((100, 100), dtype=np.uint8)

        # Apply region growing segmentation to grayscale image
        with self.assertRaises(ValueError):
            region_growing(img)

    # Handles grayscale images
    def test_large_image_segmentation_performance(self):
        # Create a large test image
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)

        # Apply region growing segmentation
        segmented = region_growing(img)

        # Check if the segmented image is not empty
        self.assertTrue(np.any(segmented != 0))
        
    # Tests if the code correctly handles images with high contrast.
    def test_handles_high_contrast_image(self):
        # Create a test image with high contrast
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = [255, 255, 255]
        img[70:90, 70:90] = [0, 0, 0]

        # Apply region growing segmentation to high contrast image
        segmented = region_growing(img)

        # Check if the segmented image is not empty
        self.assertTrue(np.any(segmented != 0))

    def test_handles_low_contrast_image(self):
        # Create a test image with low contrast
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = [127, 127, 127]

        # Apply region growing segmentation to low contrast image
        segmented = region_growing(img)

        # Check if the segmented image is not empty
        self.assertTrue(np.any(segmented != 0))
        
if __name__ == '__main__':
    unittest.main()
