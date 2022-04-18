#include <fstream>

#include "feature_extractor.hpp"

#ifndef SAVER_H
#define SAVER_H

class Saver {
private:
    std::string out_dir;
    std::ofstream mycsvfile;
    const int *counter;
public:
    Saver(const std::string out_dir_, const int *counter_);
    void close();
    void save_csv(const Frame &fr);
};

#endif