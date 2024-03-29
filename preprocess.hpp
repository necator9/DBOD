// preprocess.hpp
#include "feature_extractor.hpp"

#ifndef PREPROCESS_H
#define PREPROCESS_H

class Preproc {
private:
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;
    cv::Ptr<cv::CLAHE> clahe;
    cv::Mat f_element;
    int dilate_it, m_op_it, margin;
    cv::Size resolution;
    cv::Mat optimized_matrix;
    cv::Mat camera_matrix;
    cv::Mat dist_coefs;
    void undist(std::vector<std::vector<cv::Point>> &contours2i);
public:
    Preproc(const ConfigParser &conf);
    void prepare_mask(Frame &fr, bool test);
};

#endif
