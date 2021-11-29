#include <opencv2/opencv.hpp>
#include "capturing.hpp"


Capturing::Capturing(cv::String cam_id_, cv::Size resolution_, int fps_):
cam_id(cam_id_), resolution(resolution_), fps(fps_){
    init_camera();
};

bool Capturing::is_number(const std::string& s) {
    return !s.empty() && std::find_if(s.begin(), 
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

void Capturing::init_camera() {
    if (is_number(cam_id)) 
        cap.open(std::stoi(cam_id));
    else 
        cap.open(cam_id);

    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, resolution.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, resolution.height);

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

