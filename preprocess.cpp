#include <opencv2/opencv.hpp>
#include "config.hpp"
#include "preprocess.hpp"
#include "feature_extractor.hpp"


Preproc::Preproc() {
    pBackSub = cv::createBackgroundSubtractorMOG2(BS_HISTORY, VAR_THR, DET_SCHADOWS);
    //pBackSub = cv::createBackgroundSubtractorKNN(BS_HISTORY, 400.0, DET_SCHADOWS);
    clahe = cv::createCLAHE();
    f_element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    clahe->setClipLimit(CLAHE_LIMIT);
    clahe->setTilesGridSize(CLAHE_GRID_SZ);
};

void Preproc::prepare_mask(Frame &fr, bool test = false) {
    cv::Mat orig_frame = fr.orig_frame;
    cv::Mat fg_frame = fr.fg_frame;
    
    if (orig_frame.channels() == 3) {
        cv::cvtColor(orig_frame, orig_frame, cv::COLOR_BGR2GRAY);
    }	
        
    clahe->apply(orig_frame, orig_frame);
    pBackSub->apply(orig_frame, fg_frame);
    cv::morphologyEx(fg_frame, fg_frame, cv::MORPH_OPEN, f_element, cv::Point(-1,-1), M_OP_ITER);
    cv::threshold(fg_frame, fg_frame, 170, 255, cv::THRESH_BINARY);
    if (DIAL_ITER > 0) {
        cv::erode(fg_frame, fg_frame, f_element, cv::Point(-1,-1),  DIAL_ITER);
    }

    if (test) {
        std::vector<cv::Point> contour1 = {cv::Point(500, 300), cv::Point(500, 100), cv::Point(700, 100)};  
        // Filtering cabdidate by CA_THR - minimal area
        std::vector<cv::Point> contour2 = {cv::Point((int)(IMG_RES.width / 2), (int)(IMG_RES.height / 2)),  
                                           cv::Point((int)(IMG_RES.width / 2 + 1), (int)(IMG_RES.height / 2)),
                                           cv::Point((int)(IMG_RES.width / 2 + 1), (int)(IMG_RES.height / 2 + 1))};
        // Filtering cabdidate by MARGIN
        std::vector<cv::Point> contour3 = {cv::Point((int)(IMG_RES.width / 2), (int)(IMG_RES.height / 2)),  
                                           cv::Point((int)(IMG_RES.width), (int)(IMG_RES.height / 2)),
                                           cv::Point((int)(IMG_RES.width), (int)(IMG_RES.height / 2 + IMG_RES.height * 0.2))};
        // Filtering cabdidate by EXTENT_THR
        std::vector<cv::Point> contour4 = {cv::Point(MARGIN, IMG_RES.height - MARGIN), 
                                           cv::Point(IMG_RES.width - MARGIN, IMG_RES.height - MARGIN), 
                                           cv::Point(IMG_RES.width - MARGIN, MARGIN),
                                           cv::Point(IMG_RES.width - MARGIN - 1, MARGIN), 
                                           cv::Point(IMG_RES.width - MARGIN - 1, IMG_RES.height - MARGIN - 1)};
        
        std::vector<std::vector<cv::Point>> contours = {contour1, contour2, contour3, contour4}; 
        fr.contours = contours;
    }

    else {
        findContours(fg_frame, fr.contours, fr.hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    }

    // for(auto i = 0; i < fr.contours.size(); i++) {
    //     fr.boundRect.push_back(boundingRect(fr.contours[i]));
    //     fr.ca.push_back(contourArea(fr.contours[i]));
    // }

    for(auto i = 0; i < fr.contours.size(); i++) {
        struct BasicObjParams e = {boundingRect(fr.contours[i]), contourArea(fr.contours[i])};
        fr.basic_params.push_back(e);
    }
    
}