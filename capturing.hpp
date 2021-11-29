// capturing.hpp

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
    void init_camera();
    void get_frame(cv::Mat& frame);
    void close();
};

#endif 
