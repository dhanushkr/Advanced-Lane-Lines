# Advanced Lane Lines

![Image](test_images/test1.jpg)

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
3. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The project requires the following libraries```numpy```, ```OpenCV```, ```matplotlib```, ```Moviepy```

## Project Motivation<a name="motivation"></a>
The aim of the project is to create an pipeline to identify lane lines on the road using Camera calibration, gradients and color spaces, perspective transforms and measuring curvature and finding lanes.

## File Descriptions <a name="files"></a>
- [Camera calibration](https://github.com/dhanushkr/Advanced-Lane-Lines/blob/master/camera_calibration.ipynb) - contains the code for calibrating camera
- [calibration.p](https://github.com/dhanushkr/Advanced-Lane-Lines/blob/master/calibration.p) -  stores the camera matrix and distortion coefficients
- [Threshold Binary output](threshold_binary_image.ipynb) - applying gradients using sobel and color spaces
- [Perspective Transform](https://github.com/dhanushkr/Advanced-Lane-Lines/blob/master/perspepctive_transform.ipynb) - Obtaining perspective 
- [Pipeline](advanced_lane_lines.ipynb) - Software pipeline to detect lane lines
## Licensing, Authors, Acknowledgements<a name="licensing"></a> 
[MIT License](https://github.com/dhanushkr/Advanced-Lane-Lines/blob/master/LICENSE)


