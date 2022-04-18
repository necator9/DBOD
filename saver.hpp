#include <fstream>

#include "feature_extractor.hpp"

#ifndef SAVER_H
#define SAVER_H

class Saver {
private:
    std::string out_dir;
    std::ofstream mycsvfile;
public:
    Saver(const std::string out_dir_);
    void close();
    void save_csv(Frame &fr, cv::Mat &out_probs);
};

#endif