//
// Created by ivan on 4/6/20.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "capturing.h"

class Capturing {
    int cam_id = 0;
    cv::VideoCapture cap;

public:
    void init_camera()
    {
        cap.open(cam_id);
        if(!cap.isOpened())
            std::cerr << "Cap is not opened" << std::endl;

        // cap.set(cv::CAP_PROP_FRAME_WIDTH, 2592);
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1944);
        // cap.set(cv::CAP_PROP_FPS, 60);
    }

    void get_frame(cv::Mat& frame)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Stream has been interrupted" << std::endl;
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    }
};