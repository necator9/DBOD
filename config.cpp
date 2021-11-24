#include "config.hpp"
#include <opencv2/opencv.hpp>
#include "yaml-cpp/yaml.h"
//#include <string>


// #include <iostream>
// #include <stdlib.h>

const double RX_DEG = -20;
const double CAM_H = -3;
const cv::Size_<int> IMG_RES  = {1024, 768};
const double FL = 2.2;

const int CAM_DEV = 0;
const cv::Size_<int> RESOLUTION = {640, 480};
const int FPS = 30;

const int CLAHE_LIMIT = 3;
const cv::Size_<int> CLAHE_GRID_SZ = {8, 8};
const int  BS_HISTORY = 100;
const bool DET_SCHADOWS = true;
const int VAR_THR = 16; //MOG2 thr
const int M_OP_ITER = 3;
const int DIAL_ITER = 0;

const double CA_THR = 0.001;
const int MARGIN = 2;
const double EXTENT_THR = 0.2;
const double MAX_DIST = 25;

ConfigParser::ConfigParser(std::string yaml_path_):
yaml_path(yaml_path_) {
    parse_yaml();
};

void ConfigParser::parse_yaml() {
    YAML::Node config = YAML::LoadFile(yaml_path);
    height = config["height"].as<double>();
    angle = config["angle"].as<double>();
}

WeightsParser::WeightsParser(double height_, double angle_, std::string yaml_path_):
height(height_), angle(angle_), yaml_path(yaml_path_) {
    parse_yaml();
};

void WeightsParser::parse_yaml() {
    YAML::Node config = YAML::LoadFile(yaml_path);
    intercept = config[height][angle]["intercept"].as<std::vector<double>>();
    coef = config[height][angle]["coef"].as<std::vector<std::vector<double>>>();  
}


