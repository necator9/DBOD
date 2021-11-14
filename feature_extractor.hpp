// feature_extractor.hpp

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

// Extract object features from given bounding rectangles and contour areas
class FeatureExtraxtor {
public:
    std::vector<cv::Point> contour1 = {cv::Point(587, 476), cv::Point(584, 479), cv::Point(590, 479)};  
    std::vector<cv::Point> contour2 = {cv::Point(587, 400), cv::Point(584, 400), cv::Point(590, 300)};  
    std::vector<std::vector<cv::Point>> contours = {contour1, contour2}; 

    std::vector<std::vector<cv::Point>> contours_poly;
    std::vector<cv::Rect> boundRect;
    std::vector<double> c_a;

    std::vector<double> intrinsic_v = {602.17434328, 0, 511.32476428,
                                      0.0, 601.27444228, 334.8572872,
                                      0, 0, 1};
    cv::Mat intrinsic = cv::Mat(3, 3, CV_32SC1, intrinsic_v.data());

    double r_x_rad;               // Camera rotation angle about x axis in radians
    double cam_h;                 // Ground y coord relative to camera (cam. is origin) in meters
    cv::Size_<int> img_res;       // Image resolution (width, height) in px
    double f_l;                   // Focal length in mm
    cv::Size_<double> sens_dim;   // Camera sensor dimensions (width, height) in mm
    cv::Point_<double> cx_cy;     // Central pixel of an image in px
    double px_h_mm;               // Scaling between pixels in millimeters
    double inf = std::numeric_limits<double>::infinity();

    FeatureExtraxtor(double f_l_, cv::Size_<int> img_res_, double r_x_deg_);
    void find_basic_params();
    template<typename T_d, typename T_ybh>
    void estimate_distance(T_d &distance, const T_ybh &y_bot_hor, int col_id);
};

#endif