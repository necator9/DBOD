#include <iostream>
#include <filesystem>
#include <fstream>

#include "saver.hpp"

namespace fs = std::filesystem;

Saver::Saver(const std::string out_dir_):
    out_dir(out_dir_) {
    fs::path dir(out_dir);
    fs::path file("output.csv");
    fs::path full_csv_path = dir / file;
    mycsvfile.open(full_csv_path);
    mycsvfile << "br_x,br_y,br_w,br_h" << std::endl;
};

void Saver::close() {
    mycsvfile.close();
}

void Saver::save_csv(Frame &fr, cv::Mat &out_probs) {
    for (auto i = 0; i < fr.basic_params.size(); i++) {
        mycsvfile << fr.basic_params[i].rect.x << "," << out_probs.at<float>(i) << "," << std::endl;
    }
}
