#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

#ifndef CONFIG_H
#define CONFIG_H

// Parse yaml config file
class ConfigParser {
public:
    std::string device;
    cv::Size resolution;
    int fps;

    double height, angle, focal_length;

    std::string yaml_path;
    std::string clf;
    fs::path out_dir;

    // BS parameters
    bool shadows;
    int bs_history, var_thr;  
    // Preprocessing parameters
    int dilate_it, m_op_it, clahe_limit;  
    cv::Size clahe_grid_sz;

    // Filtering
    double cont_area_thr, extent_thr, max_distance;
    int margin;
    std::string weights;

    // Camera matrices
    cv::Size base_res;
    cv::Mat camera_matrix;
    cv::Mat dist_coefs;
    cv::Size optimized_res;
    cv::Mat optimized_matrix;

    bool save_csv;

    ConfigParser(std::string yaml_path_);
    ConfigParser();
    void parse_yaml();
    cv::Mat scale(cv::Size new_res, cv::Size base_res, cv::Mat intrinsic);

};

// Parse yaml weights file
class WeightsParser {
public:
    cv::Mat intercept, coef;
    std::string yaml_path;
    WeightsParser(std::string yaml_path_);
    WeightsParser();
    void parse_yaml();

};

#endif