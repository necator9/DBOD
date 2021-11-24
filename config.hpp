#include <opencv2/opencv.hpp>
// #include <string>


#ifndef CONFIG_H
#define CONFIG_H

extern const double RX_DEG;
extern const double CAM_H;
extern const cv::Size_<int> IMG_RES;
extern const double FL;

extern const int CAM_DEV;
extern const cv::Size_<int>  RESOLUTION;
extern const int FPS;

extern const int CLAHE_LIMIT;
extern const cv::Size_<int> CLAHE_GRID_SZ;
extern const int  BS_HISTORY;
extern const bool DET_SCHADOWS;
extern const int VAR_THR; 
extern const int M_OP_ITER;
extern const int DIAL_ITER;

extern const double CA_THR;
extern const int MARGIN;
extern const double EXTENT_THR;
extern const double MAX_DIST;

// Parse yaml config file
class ConfigParser {
public:
    std::string yaml_path;
    double height, angle;
    ConfigParser(std::string yaml_path_);
    void parse_yaml();

};

// Parse yaml weights file
class WeightsParser {
public:
    std::vector<double> intercept;
    std::vector<std::vector<double>> coef;
    double height, angle;
    std::string yaml_path;

    WeightsParser(double height_, double angle_, std::string yaml_path_);
    void parse_yaml();

};

#endif