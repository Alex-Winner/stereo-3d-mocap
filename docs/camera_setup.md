# Camera Calibration Setup Instructions

This document provides step-by-step instructions for setting up your cameras and performing camera calibration using OpenCV. Calibration is essential for various computer vision tasks such as 3D reconstruction and object tracking.

## Table of Contents

1. [Hardware Setup](#hardware-setup)
   - [Camera Placement](#camera-placement)
   - [Checkerboard Preparation](#checkerboard-preparation)

2. [Software Installation](#software-installation)
   - [Python and OpenCV](#python-and-opencv)

3. [Camera Calibration](#camera-calibration)
   - [Single Camera Calibration](#single-camera-calibration)
   - [Stereo Camera Calibration](#stereo-camera-calibration)

4. [Result Verification](#result-verification)

## Hardware Setup

### Camera Placement

1. Place the cameras in the desired positions and angles for calibration.
2. Ensure that the cameras have a clear view of the calibration pattern (checkerboard) from different angles.
3. Make sure the cameras are securely mounted and stable during the calibration process.

### Checkerboard Preparation

1. Prepare a checkerboard pattern with known square size (e.g., 20 mm).
2. Ensure that the checkerboard has the specified number of rows and columns (e.g., 4 rows and 7 columns).
3. Print the checkerboard pattern on a flat, non-reflective surface or attach it to a rigid board.
4. Keep the checkerboard clean and free from any damage or deformation.

## Software Installation

### Python and OpenCV

1. Install Python on your computer if it is not already installed. You can download Python from [python.org](https://www.python.org/downloads/).

2. Install OpenCV, a popular computer vision library, using the following command:

   ```bash
   pip install opencv-python

## Single Camera Calibration

1. Capture multiple images of the checkerboard from different angles using the single camera you want to calibrate. Ensure good coverage of the checkerboard in each image.

2. Create a Python script for single camera calibration. You can refer to sample scripts or use libraries like OpenCV for this purpose.

3. Execute the script, which will save the calibration results and undistorted images.

## Stereo Camera Calibration
1. Capture stereo image pairs of the checkerboard from different angles using both the left and right cameras simultaneously.

2. Create a Python script for stereo camera calibration using libraries like OpenCV.

3. Execute the script, which will save the stereo calibration results and rectify the stereo images.

## Result Verification

After completing the calibration process, it's essential to verify the accuracy of the calibration results. You can do this by:

1. Capturing new images and using the calibration parameters to undistort them.
2. Measuring the accuracy of the undistorted images, ensuring that straight lines remain straight and known distances are preserved.
By following these instructions, you can set up and perform camera calibration effectively for your computer vision applications.

For troubleshooting and additional guidance, refer to the OpenCV documentation and community resources.