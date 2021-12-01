// feature_extractor.hpp
#include "config.hpp"
#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

template <typename T> std::vector<T> flatten(const std::vector<std::vector<T>>& v);

struct BasicObjParams {
    cv::Rect rect;
    double ca;
    double rw_d;
    double rw_h;
    double rw_w;
    double rw_ca;
    };

class Frame{
public:
    cv::Mat orig_frame, fg_frame;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Rect> boundRect;
    std::vector<double> ca;
    std::vector<BasicObjParams> basic_params;
    // Frame();
    friend std::ostream& operator<<(std::ostream& os, const Frame& fr);
};


// Extract object features from given bounding rectangles and contour areas
class FeatureExtraxtor {
public:
    cv::Mat intrinsic = cv::Mat_<double>(3, 3);
    cv::Mat intrinsic_inv = cv::Mat_<double>(3, 3); 

    // Rotation matrix around the X axis
    cv::Mat rot_x_mtx = cv::Mat_<double>(4, 4);
    cv::Mat rot_x_mtx_inv = cv::Mat_<double>(4, 4);

    double rx_rad, rx_deg;       // Camera rotation angle about x axis in radians
    double cam_h;                 // Ground y coord relative to camera (cam. is origin) in meters
    cv::Size_<int> img_res;       // Image resolution (width, height) in px
    double fl;                    // Focal length in mm
    cv::Size_<double> sens_dim;   // Camera sensor dimensions (width, height) in mm
    cv::Point_<double> cx_cy;     // Central pixel of an image in px
    double px_h_mm;               // Scaling between pixels in millimeters
    double inf = std::numeric_limits<double>::infinity();
    int n_obj;

    FeatureExtraxtor(double fl_, double cam_h_, cv::Size_<int> img_res_, double r_x_deg_, cv::Mat intrinsic);
    FeatureExtraxtor(const ConfigParser& conf);
    void init();
    void extract_features(Frame& fr);
    void compose_mtx(Frame& fr, cv::Mat& boundRect_arr, cv::Mat& ca_px);
    void decompose_mtx(Frame& fr, cv::Mat& features);
    void estimate_distance(cv::Mat& distance, const cv::Mat& ang_y_bot_to_hor);
    void estimate_height(cv::Mat& height, const cv::Mat& distance, const cv::Mat& ang_y_bot_top_to_hor);
    void estimate_3d_coordinates(cv::Mat& rw_coords, const cv::Mat& px_x_lr, const cv::Mat& rw_distance);
};


class Classifier {
public:
    std::vector<double> polynomialFeatures(const std::vector<double>& input, unsigned int degree, bool interaction_only, bool include_bias);
    std::vector<std::vector<double>> matMul(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B);
    static std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>> data);
    static double myproduct (double x, double* y);
    void classify(Frame& fr,  cv::Mat& out_probs);
    Classifier(const std::string& weight_path);
    WeightsParser weights;
};





#endif
