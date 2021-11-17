//
// Created by ivan on 4/6/20.
//
#include <opencv2/opencv.hpp>
#include "capturing.hpp"


Capturing::Capturing(int cam_id_, int width_, int height_, int fps_):
cam_id(cam_id_), width(width_), height(height_), fps(fps_){
    init_camera();
};

void Capturing::init_camera() {
    cap.open(cam_id);
    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    if (!cap.isOpened()) std::cerr << "Cap is not opened" << std::endl;
}

void Capturing::get_frame(cv::Mat& frame) {
    cap >> frame;
    if (frame.empty()) std::cerr << "Stream has been interrupted" << std::endl;
}

void Capturing::close() {
    cap.release();
    std::cout << "Capturing device closed" << std::endl;
}

