#include <iostream>
#include <experimental/filesystem>
#include <fstream>

#include "saver.hpp"

namespace fs = std::experimental::filesystem;

Saver::Saver(const fs::path out_dir_, const bool save_, const int *counter_):
    out_dir(out_dir_), save(save_), counter(counter_) {
    if (save) {
        fs::path file("output.csv");
        fs::path full_csv_path = out_dir / file;
        mycsvfile.open(full_csv_path.string());
        mycsvfile << "img,o_num,rw_w,rw_h,rw_ca,rw_z,rw_x,ca,x,y,w,h,o_prob,o_class,     -av_bin" << std::endl;
    }
};

void Saver::close() {
    if (save) {
        mycsvfile.close();
    }
}

void Saver::save_csv(const Frame &fr) {
    if (save) {
        for (auto i = 0; i < fr.basic_params.size(); i++) {
            mycsvfile << *counter << "," << i << "," << fr.basic_params[i].rw_w << "," << fr.basic_params[i].rw_h << "," << 
            fr.basic_params[i].rw_ca << "," << fr.basic_params[i].rw_d << "," << fr.basic_params[i].rw_xc << "," <<   
            fr.basic_params[i].ca << "," << fr.basic_params[i].rect.x << "," << fr.basic_params[i].rect.y << "," << 
            fr.basic_params[i].rect.width << "," << fr.basic_params[i].rect.height << "," << fr.prob[i] << "," <<
            fr.o_class[i] << std::endl;
        }
    }
}
