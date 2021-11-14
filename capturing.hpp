// capturing.hpp

#ifndef CAPTURING_CAPTURING_H
#define CAPTURING_CAPTURING_H



class Capturing {
private:
    cv::VideoCapture cap;
    int cam_id, width, height, fps;
public:
    Capturing(int cam_id_, int width_, int height_, int fps);
    void init_camera();
    void get_frame(cv::Mat& frame);
};

#endif 
