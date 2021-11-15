#include <opencv2/opencv.hpp>
#include <math.h>
#include <limits>
#include "feature_extractor.hpp"


FeatureExtraxtor::FeatureExtraxtor(double fl_, double cam_h_, cv::Size_<int> img_res_, double rx_deg_):
fl(fl_), cam_h(cam_h_), img_res(img_res_){
    sens_dim.width = fl * img_res.width / intrinsic.at<double>(0, 0);   // / fx
    sens_dim.height = fl * img_res.height / intrinsic.at<double>(1, 1);   // / fy
    cx_cy = {intrinsic.at<double>(0, 2), intrinsic.at<double>(1, 2)};    
    px_h_mm = sens_dim.height / (fl * img_res.height);
    rx_rad = rx_deg_ * (M_PI / 180);
};

void FeatureExtraxtor::find_basic_params(){
    for(size_t i = 0; i < contours.size(); i++){
        //approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect.push_back(boundingRect(contours[i]));
        ca.push_back(contourArea(contours[i]));
    }

    // Transform bounding rectangles to required shape
    // Important! Reverse the y coordinates of bound.rect. along y axis before transformations (self.img_res[1] - y)
    std::vector<std::vector<double>> px_y_bottom_top, y_bottom_top_to_hor;
    double px_y_bottom_top_p1, px_y_bottom_top_p2;
    for(size_t i = 0; i < contours.size(); i++){
        px_y_bottom_top_p1 = (double)img_res.height - boundRect[i].br().y;
        px_y_bottom_top_p2 = (double)img_res.height - boundRect[i].y;
        px_y_bottom_top.push_back(std::vector<double> {px_y_bottom_top_p1, px_y_bottom_top_p2});
        // Distances from vertices to img center (horizon) along y axis, in px
        y_bottom_top_to_hor.push_back(std::vector<double> {cx_cy.y - px_y_bottom_top_p1, cx_cy.y - px_y_bottom_top_p2});  
        //Convert to mm and find angle between object pixel and central image pixel along y axis
        y_bottom_top_to_hor[i][0] = atan(y_bottom_top_to_hor[i][0] * px_h_mm); 
        y_bottom_top_to_hor[i][1] = atan(y_bottom_top_to_hor[i][1] * px_h_mm); 
    }

    // Find object distance in real world
    std::vector<double> rw_distance(contours.size());
    estimate_distance(rw_distance, y_bottom_top_to_hor, 0);    // Passed arg is angles to bottom vertices

    // Find object height in real world
    std::vector<double> rw_height(contours.size());
    estimate_height(rw_height, rw_distance, y_bottom_top_to_hor);

}


// Estimate distance to the bottom pixel of a bounding rectangle. Based on assumption that object is aligned with the
// ground surface. Calculation uses angle between vertex and optical center along vertical axis
template<typename T_1d, typename T_2d>
void FeatureExtraxtor::estimate_distance(T_1d &distance, const T_2d &y_bot_hor, const int col_id){
    double deg;
    for (auto i = 0; i < distance.size(); i++){
        deg = y_bot_hor[i][col_id] - rx_rad;
        distance[i] = abs(cam_h) / (deg >= 0 ? tan(deg) : inf);
    }
}

// Estimate height of object in real world
template<typename T_1d, typename T_2d>
void FeatureExtraxtor::estimate_height(T_1d &height, const T_1d &distance, const T_2d &ang_y_bot_top_to_hor) {
    double angle_between_pixels, gamma, beta;
    for (auto i = 0; i < distance.size(); i++){
        angle_between_pixels = abs(ang_y_bot_top_to_hor[i][0] - ang_y_bot_top_to_hor[i][1]);
        gamma = atan(distance[i] * 1 / abs(cam_h));
        beta = M_PI - angle_between_pixels - gamma;
        height[i] = hypot(abs(cam_h), distance[i]) * sin(angle_between_pixels) / sin(beta);
    }
}
