#include "config.hpp"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

template<>
struct YAML::convert<cv::Size> {
  static bool decode(const Node& node, cv::Size& sz) {
    if(!node.IsSequence() || node.size() != 2) {
      return false;
    }
    sz.width = node[0].as<int>();
    sz.height = node[1].as<int>();

    return true;
  }
};

// Parse 2D matrix
template<> 
struct YAML::convert<cv::Mat> {
  static bool decode(const Node& node, cv::Mat& m) {
    if(!node.IsSequence() || !node[0].IsSequence()) {
        return false;
    }

    for (auto r : node) {
        std::vector<double> rv;
        for (auto c : r) {
            rv.push_back(c.as<double>());
        }
        m.push_back(rv);
    }
    m = m.reshape(0, node.size());

    return true;
  }
};



ConfigParser::ConfigParser(std::string yaml_path_):
yaml_path(yaml_path_) {
    parse_yaml();
};

ConfigParser::ConfigParser() {
};

void ConfigParser::parse_yaml() {
    YAML::Node config = YAML::LoadFile(yaml_path);
    device = cv::String(config["device"].as<std::string>());
    resolution = config["resolution"].as<cv::Size>();
    fps = config["fps"].as<int>();

    // out_dir = config["out_dir"].as<std::string>();
    out_dir = config["out_dir"].as<std::string>();

    height = config["height"].as<double>();
    angle = config["angle"].as<double>();
    focal_length = config["focal_length"].as<double>();

    cont_area_thr = config["cont_area_thr"].as<double>();
    margin = config["margin"].as<int>();
    extent_thr = config["extent_thr"].as<double>();
    max_distance = config["max_distance"].as<double>();

    bs_history = config["bs_history"].as<int>();
    var_thr = config["var_thr"].as<int>();
    shadows = config["shadows"].as<bool>();

    clahe_limit = config["clahe_limit"].as<int>();
    clahe_grid_sz = config["clahe_grid_sz"].as<cv::Size>();
    dilate_it = config["dilate_it"].as<int>();
    m_op_it = config["m_op_it"].as<int>();

    weights = config["weights"].as<std::string>();
    base_res = config["base_res"].as<cv::Size>();
    camera_matrix = scale(resolution, base_res, config["camera_matrix"].as<cv::Mat>());
    dist_coefs = config["dist_coefs"].as<cv::Mat>();
    optimized_res = config["optimized_res"].as<cv::Size>();
    optimized_matrix = config["optimized_matrix"].as<cv::Mat>();

    save_csv = config["save_csv"].as<bool>();
}

// Scale intrinsic matrix according to the current capturing resolution 
cv::Mat ConfigParser::scale(cv::Size new_res, cv::Size base_res, cv::Mat intrinsic) {
    double scale_fw = base_res.width / new_res.width;
    double scale_fh = base_res.height / new_res.height;

    if (scale_fw != scale_fh) 
        std::cout << "WARNING! Scaling is not proportional: " <<  scale_fw << " != " << scale_fh << std::endl;
    
    intrinsic.row(0) /= scale_fw;
    intrinsic.row(1) /= scale_fh;

    return intrinsic;
}


WeightsParser::WeightsParser(std::string yaml_path_):
yaml_path(yaml_path_) {
    parse_yaml();
};

WeightsParser::WeightsParser() {
};

void WeightsParser::parse_yaml() {
    cv::FileStorage fs;
    fs.open(yaml_path, cv::FileStorage::READ);
    fs["coef"] >> coef;                                 
    fs["intercept"] >> intercept;
    intercept = intercept.t();
}



