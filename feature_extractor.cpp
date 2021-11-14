#include <opencv2/opencv.hpp>
#include <math.h>
#include <limits>

#include "feature_extractor.hpp"


FeatureExtraxtor::FeatureExtraxtor(double f_l_, cv::Size_<int> img_res_, double r_x_deg_):
f_l(f_l_), img_res(img_res_){
    sens_dim.width = f_l * img_res.width / intrinsic.at<double>(0, 0);   // / fx
    sens_dim.height = f_l * img_res.height / intrinsic.at<double>(1, 1);   // / fy
    cx_cy = {intrinsic.at<double>(0, 2), intrinsic.at<double>(1, 2)};    
    px_h_mm = sens_dim.height / (f_l * img_res.height);
    r_x_rad = r_x_deg_ * (M_PI / 180);
};

void FeatureExtraxtor::find_basic_params(){
    for(size_t i = 0; i < contours.size(); i++)
    {
        //approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect.push_back(boundingRect(contours[i]));
        c_a.push_back(contourArea(contours[i]));
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
    
        std::vector<double> distance(contours.size());
        estimate_distance(distance, y_bottom_top_to_hor, 0);    // Passed arg is angles to bottom vertices
}

// Find object distance in real world
template<typename T_d, typename T_ybh>
void FeatureExtraxtor::estimate_distance(T_d &distance, const T_ybh &y_bot_hor, const int col_id) {
    double deg;
    for (auto i = 0; i < distance.size(); i++) {
        deg = y_bot_hor[i][col_id] - r_x_rad;
        distance[i] = abs(cam_h) / (deg >= 0 ? tan(deg) : inf);
    }

}