# Image-Segmentation-Methods

This repository contains implementations of various image segmentation methods. Image segmentation is a fundamental task in computer vision that involves dividing an image into meaningful and distinct regions. Each method in this repository provides a different approach to achieve image segmentation.

  
The images used come from [The PASCAL Visual Object Classes Challenge](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/segexamples/index.html).

## Implemented Methods

### 1- Region Growing

Segments an image based on the similarity of pixel regions.  
Algorithm: 
- Choose a seed pixel. 
- Add neighboring pixels to the region if their intensity is similar to the seed pixel.
- Repeat until the region stops growing.

### 2- Thresholding (Adaptive Fisher/Otsu)  
- Segments an image by setting a threshold dynamically based on pixel intensities.
- Utilizes adaptive thresholding methods like Fisher or Otsu to determine optimal thresholds.

### 3- Color-based Segmentation (K-means Clustering)

- Segments an image based on color information.
- Applies **K-means** clustering to group pixels with similar colors into segments.

### 4- Edge Detection (Canny, Sobel)

- Segments an image by detecting edges.
- Utilizes edge detection algorithms such as Canny and Sobel to identify boundaries between different regions.

### 5- Active Contour (Snake)

- Segments an image using active contours.
- Snakes are deformable contours that move toward object boundaries based on energy minimization.

### 6- Semantic Segmentation (Convolutional Neural Network - CNN)
- Segments an image into different semantic classes.
- Utilizes a Convolutional Neural Network (CNN) trained for semantic segmentation tasks.

### 7- Instance Segmentation (Mask R-CNN)

- Segments individual instances of objects in an image.
- Utilizes the Mask R-CNN architecture for accurate instance-level segmentation.

# Repository Structure

- __docs__ : Directory containing example images. 
- __examples__ :  
- __methods__ : Directory containing implementation of segmentation methods.  
  | __region-based__ : Contains scripts to perform region-based segmentation.

- __tests__ : Directory containing the unit tests of the functions used.
- __README.md__ :  Project README file.
- __requirements.txt__ : Required Python packages.

# Getting Started

To clone this repository :  
```bash
git clone https://github.com/ElBaron86/Image-Segmentation-Methods.git
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions to this project are welcome! If you have ideas for improvements or new segmentation methods, please open an issue or submit a pull request.  

  

  


  



`Feel free to customize this README to better fit your project's specifics. Add more details, instructions, or information as needed.`