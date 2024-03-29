// capturing.hpp
#include "config.hpp"

#ifndef CAPTURING_CAPTURING_H
#define CAPTURING_CAPTURING_H

class Capturing {
private:
    cv::VideoCapture cap;
    int fps;
    cv::String cam_id;
    cv::Size resolution;
    bool is_number(const std::string& s);
public:
    Capturing(cv::String cam_id_, cv::Size resolution_, int fps);
    Capturing(const ConfigParser &conf);
    void init_camera();
    bool get_frame(cv::Mat& frame);
    void close();
};

#endif 
