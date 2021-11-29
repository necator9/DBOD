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
public:
    Preproc(const ConfigParser &conf);
    void prepare_mask(Frame &fr, bool test);
};

#endif
