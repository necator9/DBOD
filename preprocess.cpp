#include <opencv2/opencv.hpp>
#include "config.hpp"
#include "preprocess.hpp"


Preproc::Preproc() {
    pBackSub = cv::createBackgroundSubtractorMOG2(BS_HISTORY, VAR_THR, DET_SCHADOWS);
    //pBackSub = cv::createBackgroundSubtractorKNN(BS_HISTORY, 400.0, DET_SCHADOWS);
    clahe = cv::createCLAHE();
    f_element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    clahe->setClipLimit(CLAHE_LIMIT);
    clahe->setTilesGridSize(CLAHE_GRID_SZ);
};

void Preproc::prepare_mask(cv::Mat& orig_frame, cv::Mat& fg_frame) {
   // cv::cvtColor(orig_frame, orig_frame, cv::COLOR_BGR2GRAY);
    clahe->apply(orig_frame, orig_frame);
    pBackSub->apply(orig_frame, fg_frame);
    cv::morphologyEx(fg_frame, fg_frame, cv::MORPH_OPEN, f_element,
            cv::Point(-1,-1), M_OP_ITER);
    cv::threshold(fg_frame, fg_frame, 170, 255, cv::THRESH_BINARY);
    if (DIAL_ITER > 0) cv::erode(fg_frame, fg_frame, f_element, cv::Point(-1,-1),  DIAL_ITER);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(fg_frame, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
}