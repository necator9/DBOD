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
       // Rotation matrix around the X axis
    rot_x_mtx = (cv::Mat_<double>(4, 4) <<
        1,          0,           0, 0,
        0, cos(rx_rad), -sin(rx_rad), 0,
        0, sin(rx_rad),  cos(rx_rad), 0,
        0,          0,           0, 1);
    rot_x_mtx_inv = rot_x_mtx.inv();

    for(size_t i = 0; i < contours.size(); i++){
        //approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect.push_back(boundingRect(contours[i]));
        ca.push_back(contourArea(contours[i]));
    }
};

void FeatureExtraxtor::find_basic_params(){
    // Transform bounding rectangles to required shape
    // Important! Reverse the y coordinates of bound.rect. along y axis before transformations (self.img_res[1] - y)
    std::vector<std::vector<int>> px_y_bottom_top;
    std::vector<std::vector<double>> y_bottom_top_to_hor;
    int px_y_bottom_top_p1, px_y_bottom_top_p2;
    for(size_t i = 0; i < contours.size(); i++){
        px_y_bottom_top_p1 = img_res.height - boundRect[i].br().y;
        px_y_bottom_top_p2 = img_res.height - boundRect[i].y;
        px_y_bottom_top.push_back(std::vector<int> {px_y_bottom_top_p1, px_y_bottom_top_p2});
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
    
    //         # * Transform bounding rectangles to required shape
        // # Build a single array from left and right rects' coords to compute within single vectorized transformation
  //  std::vector<double> px_x_l(contours.size()), px_x_r(contours.size());  // Left and right rectangle coordinates along x-axis
    cv::Mat px_x_l = cv::Mat_<int>::ones(contours.size(), 3);
    // Iterate over all pixels of the image
    for(int r = 0; r < px_x_l.rows; r++){
        int* ptr = px_x_l.ptr<int>(r);  // Obtain a pointer to the beginning of row r
        ptr[0] = boundRect[r].x;        // Left bottom coord
        ptr[1] = px_y_bottom_top[r][0];
        ptr[2] = 1;                     // To hom. coordinates
       // std::cout << px_x_l.at<int>(r, 0) << "  " << px_x_l.at<int>(r, 1) << "  " << px_x_l.at<int>(r, 2) << std::endl;
    }

    cv::Mat px_x_r = cv::Mat_<int>::ones(contours.size(), 3);
    for(int r = 0; r < px_x_r.rows; r++){
        int* ptr = px_x_r.ptr<int>(r);    // Obtain a pointer to the beginning of row r
        ptr[0] = boundRect[r].br().x;     // Right bottom coord
        ptr[1] = px_y_bottom_top[r][0];
        ptr[2] = 1;
       // std::cout << px_x_r.at<int>(r, 0) << "  " << px_x_r.at<int>(r, 1) << "  " << px_x_r.at<int>(r, 2) << std::endl;
    }

    // Z cam is a scaling factor which is needed for 3D reconstruction
    std::vector<double> z_cam_coords(contours.size());
    for(auto r = 0; r < z_cam_coords.size(); r++){
        z_cam_coords[r] = cam_h * sin(rx_rad) + rw_distance[r] * cos(rx_rad);
        //std::cout << rw_distance[r] << std::endl;
    }

    cv::Mat cam_xl_yb_h = cv::Mat_<double>(contours.size(), 3);
    for(int r = 0; r < cam_xl_yb_h.rows; r++){
        double* ptr_cam_xl_yb_h = cam_xl_yb_h.ptr<double>(r);
        int* ptr_px_x_l = px_x_l.ptr<int>(r);
        for(int c = 0; c < cam_xl_yb_h.cols; c++){
            ptr_cam_xl_yb_h[c] = ptr_px_x_l[c] * z_cam_coords[r];
            // std::cout << ptr_cam_xl_yb_h[c] << std::endl;
        }
    }

    cv::Mat cam_xr_yb_h = cv::Mat_<double>(contours.size(), 3);
    for(int r = 0; r < cam_xr_yb_h.rows; r++){
        double* ptr_cam_xr_yb_h = cam_xr_yb_h.ptr<double>(r);
        int* ptr_px_x_r = px_x_r.ptr<int>(r);
        for(int c = 0; c < cam_xr_yb_h.cols; c++){
            ptr_cam_xr_yb_h[c] = ptr_px_x_r[c] * z_cam_coords[r];
            // std::cout << ptr_cam_xr_yb_h[c] << std::endl;
        }
    }

    cv::Mat camera_coords_l = intrinsic_inv * cam_xl_yb_h.t();
    cv::Mat camera_coords_r = intrinsic_inv * cam_xr_yb_h.t();

    
    // cv::Mat camera_coords_lh = cv::Mat_<double>(4, 4);
    // cv::Mat camera_coords_rh = cv::Mat_<double>(4, 4);

    // cv::convertPointsToHomogeneous(camera_coords_l, camera_coords_lh);
    // cv::convertPointsToHomogeneous(camera_coords_r, camera_coords_rh);

    cv::Mat h_row = cv::Mat_<double>::ones(1, camera_coords_l.cols); 
    camera_coords_l.push_back(h_row);  
    camera_coords_r.push_back(h_row); 

    cv::Mat rw_coords_l = (rot_x_mtx_inv * camera_coords_l).t();
    cv::Mat rw_coords_r = (rot_x_mtx_inv * camera_coords_r).t();

    // std::cout << format(rw_coords_l, cv::Formatter::FMT_NUMPY ) << std::endl << std::endl;
    // std::cout << format(rw_coords_r, cv::Formatter::FMT_NUMPY ) << std::endl << std::endl;


    //**************************************************




}


// Estimate distance to the bottom pixel of a bounding rectangle. Based on assumption that object is aligned with the
// ground surface. Calculation uses angle between vertex and optical center along vertical axis
template<typename T_1d, typename T_2d>
void FeatureExtraxtor::estimate_distance(T_1d &distance, const T_2d &y_bot_hor, const int col_id){
    double deg;
    for (auto i = 0; i < distance.size(); i++){
        deg = y_bot_hor[i][col_id] - rx_rad;
        // std::cout << y_bot_hor[i][col_id] << rx_rad << std::endl;
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
