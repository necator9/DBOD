// preprocess.hpp

#ifndef PREPROCESS_H
#define PREPROCESS_H

class Preproc {
private:
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;
    cv::Ptr<cv::CLAHE> clahe;
    cv::Mat f_element;
public:
    Preproc();
    void prepare_mask(cv::Mat& orig_frame, cv::Mat& fg_frame);
};

#endif
