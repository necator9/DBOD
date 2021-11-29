#include "config.hpp"
#include <opencv2/opencv.hpp>
#include "yaml-cpp/yaml.h"
#include <opencv2/opencv.hpp>


namespace YAML {
template<>
struct convert<cv::Size> {
  static Node encode(const cv::Size& sz) {
    Node node;
    node.push_back(sz.width);
    node.push_back(sz.height);

    return node;
  }

  static bool decode(const Node& node, cv::Size& sz) {
    if(!node.IsSequence() || node.size() != 2) {
      return false;
    }
    sz.width = node[0].as<int>();
    sz.height = node[1].as<int>();

    return true;
  }
};
}


ConfigParser::ConfigParser(std::string yaml_path_):
yaml_path(yaml_path_) {
    parse_yaml();
};

void ConfigParser::parse_yaml() {
    YAML::Node config = YAML::LoadFile(yaml_path);
    device = cv::String(config["device"].as<std::string>());
    resolution = config["resolution"].as<cv::Size>();
    fps = config["fps"].as<int>();

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



