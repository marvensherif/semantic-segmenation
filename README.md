# Semantic Segmentation Project Documentation

This document details two approaches for semantic segmentation:

1. **Classical Image Processing Techniques** (OpenCV-based)
2. **SegFormer Transformer Model** (Hugging Face implementation)

## Task 1: Classical Segmentation Pipeline

### Overview

The traditional computer vision approach for barcode segmentation involves sequential image processing operations. These steps include noise reduction, thresholding, edge detection, morphological operations, and contour processing.


    A[Original Image] --> B[Gaussian Blur]
    B --> C[Otsu's Thresholding]
    C --> D[Canny Edge Detection]
    D --> E[Morphological Operations]
    E --> F[Contour Filling]
    F --> G[Final Segmentation] 

### Key Operations

1. **Noise Reduction**
   - The image is smoothed using `cv2.GaussianBlur()` with a 5x5 kernel, preserving edges while reducing noise.

2. **Adaptive Thresholding**
   - `cv2.THRESH_OTSU` automatically determines the optimal threshold value to segment the image.

3. **Edge Detection**
   - `cv2.Canny()` is used with thresholds `(50,150)` to identify strong and weak edges in the image.

4. **Morphological Processing**
   - **Dilation**: Performed for 2 iterations to connect broken edges.
   - **Closing**: Applied for 2 iterations to fill small holes within the image.

5. **Contour Processing**
   - `cv2.findContours()` with the `RETR_EXTERNAL` flag is used to detect outer boundaries of the segmented object.

**Tested on**: Barcode images to segment and identify the barcodes effectively.


# Task 2: SegFormer Transformer Model Implementation

## Overview
SegFormer is a state-of-the-art transformer-based architecture for semantic segmentation. It is fine-tuned on the ADE20K dataset, which contains 150 semantic categories. The architecture features a lightweight MLP decoder and a Mix Transformer (MiT) backbone.

## Model Architecture
- **Backbone:** Mix Transformer (MiT)
- **Decoder:** Lightweight MLP decoder
- **Input Resolution:** 640x640
- **Pretrained on:** ADE20K dataset (150 semantic categories)





