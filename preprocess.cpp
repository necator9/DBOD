#include <opencv2/opencv.hpp>
#include "config.hpp"
#include "preprocess.hpp"
#include "feature_extractor.hpp"


Preproc::Preproc(const ConfigParser &conf) {
    dilate_it = conf.dilate_it;
    m_op_it = conf.m_op_it;
    pBackSub = cv::createBackgroundSubtractorMOG2(conf.bs_history, conf.var_thr, conf.shadows);
    //pBackSub = cv::createBackgroundSubtractorKNN(BS_HISTORY, 400.0, DET_SCHADOWS);
    clahe = cv::createCLAHE();
    f_element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    clahe->setClipLimit(conf.clahe_limit);
    clahe->setTilesGridSize(conf.clahe_grid_sz);
    resolution = conf.resolution;
    margin = conf.margin;
    optimized_matrix = conf.optimized_matrix;
    camera_matrix = conf.camera_matrix;
    dist_coefs = conf.dist_coefs;
};

void Preproc::prepare_mask(Frame &fr, bool test = false) {
    cv::Mat orig_frame = fr.orig_frame;
    cv::Mat fg_frame = fr.fg_frame;
    
    if (orig_frame.channels() == 3) {
        cv::cvtColor(orig_frame, orig_frame, cv::COLOR_BGR2GRAY);
    }
        
    clahe->apply(orig_frame, orig_frame);
    pBackSub->apply(orig_frame, fg_frame);
    cv::morphologyEx(fg_frame, fg_frame, cv::MORPH_OPEN, f_element, cv::Point(-1,-1), m_op_it);
    cv::threshold(fg_frame, fg_frame, 170, 255, cv::THRESH_BINARY);
    if (dilate_it > 0) {
        cv::erode(fg_frame, fg_frame, f_element, cv::Point(-1,-1),  dilate_it);
    }
    
    if (test) {
        std::vector<cv::Point> contour1 = {cv::Point(500, 300), cv::Point(500, 100), cv::Point(700, 100)};  
        // Filtering cabdidate by CA_THR - minimal area
        std::vector<cv::Point> contour2 = {cv::Point((int)(resolution.width / 2), (int)(resolution.height / 2)),  
                                           cv::Point((int)(resolution.width / 2 + 1), (int)(resolution.height / 2)),
                                           cv::Point((int)(resolution.width / 2 + 1), (int)(resolution.height / 2 + 1))};
        // Filtering cabdidate by MARGIN
        std::vector<cv::Point> contour3 = {cv::Point((int)(resolution.width / 2), (int)(resolution.height / 2)),  
                                           cv::Point((int)(resolution.width), (int)(resolution.height / 2)),
                                           cv::Point((int)(resolution.width), (int)(resolution.height / 2 + resolution.height * 0.2))};
        // Filtering cabdidate by EXTENT_THR
        std::vector<cv::Point> contour4 = {cv::Point(margin, resolution.height - margin), 
                                           cv::Point(resolution.width - margin, resolution.height - margin), 
                                           cv::Point(resolution.width - margin, margin),
                                           cv::Point(resolution.width - margin - 1, margin), 
                                           cv::Point(resolution.width - margin - 1, resolution.height - margin - 1)};
        
        std::vector<std::vector<cv::Point>> contours = {contour1, contour2, contour3, contour4}; 
        fr.contours = contours;
    }

    else {
        findContours(fg_frame, fr.contours, fr.hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    }

    // Undistort points
    undist(fr.contours);

    // for(auto i = 0; i < fr.contours.size(); i++) {
    //     fr.boundRect.push_back(boundingRect(fr.contours[i]));
    //     fr.ca.push_back(contourArea(fr.contours[i]));
    // }

    for(auto i = 0; i < fr.contours.size(); i++) {
        struct BasicObjParams e = {boundingRect(fr.contours[i]), contourArea(fr.contours[i])};
        fr.basic_params.push_back(e);
    }
    
}

void Preproc::undist(std::vector<std::vector<cv::Point>> &contours2i) {
    // Convert to points to float -> undistort -> convert to int -> assign to the original vector  
    std::vector<std::vector<cv::Point2f>> contours2f;
    for (auto i = 0; i < contours2i.size(); i++) {
        std::vector<cv::Point2f> contour2f;
        std::transform(contours2i[i].begin(), contours2i[i].end(), std::back_inserter(contour2f), [](const cv::Point& p) { return (cv::Point2f)p; });
        cv::undistortPoints(contour2f, contour2f, camera_matrix, dist_coefs, cv::Mat(), optimized_matrix);
        std::transform(contour2f.begin(), contour2f.end(), std::back_inserter(contours2i[i]), [](const cv::Point2f& p) { return (cv::Point)p; });
    }
}