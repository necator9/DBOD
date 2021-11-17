# Dimentional based object detection algorithm (DBOD)
## This is a C++ version of DBOD algorithm [written in Python](https://github.com/necator9/detection_method)
The current repository is an implementation of detection method for low-performance Linux single-board computers.
The method is used for detection of pedestrians, cyclists and vehicles in city environment.
The method is based on analysis of geometrical object features in a foreground mask. The foreground mask is obtained using background subtraction algorithm.
Classification is performed using logistic regression classifier.
Implementation of the method is based on the publication [“Fast Object Detection Using Dimensional Based Features for Public Street Environments”](https://www.mdpi.com/2624-6511/3/1/6).

## Prerequisites
The method can be used **only** when following conditions are satisfied:
1) Known intrinsic and extrinsic (angle about X axis and height of installation) camera parameters.
2) The camera is mounted on a static object.
3) The [trained classifier](https://github.com/necator9/model_training) for a particular usage scenario. The training uses 3D object models and camera parameters on input.