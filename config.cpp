#include "config.hpp"
#include <opencv2/opencv.hpp>

const int CAM_DEV = 0;
const cv::Size_<int> RESOLUTION = {640, 480};
const int FPS = 30;

const int CLAHE_LIMIT = 3;
const cv::Size_<int> CLAHE_GRID_SZ = {8, 8};
const int  BS_HISTORY = 100;
const bool DET_SCHADOWS = true;
const int VAR_THR = 16; //MOG2 thr
const int M_OP_ITER = 3;
const int DIAL_ITER = 0;