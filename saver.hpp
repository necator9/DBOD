#include <fstream>

#include "feature_extractor.hpp"

#ifndef SAVER_H
#define SAVER_H

class Saver {
private:
    fs::path out_dir;
    std::ofstream mycsvfile;
    const int *counter;
    const bool save;
public:
    Saver(const fs::path out_dir_, const bool save_, const int *counter_);
    void close();
    void save_csv(const Frame &fr);
};

#endif