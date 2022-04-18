#include <iostream>
#include <filesystem>
#include <fstream>

#include "saver.hpp"

namespace fs = std::filesystem;

Saver::Saver(const std::string out_dir_, const int *counter_):
    out_dir(out_dir_), counter(counter_) {
    fs::path dir(out_dir);
    fs::path file("output.csv");
    fs::path full_csv_path = dir / file;
    mycsvfile.open(full_csv_path);
    mycsvfile << "img,o_num,rw_w,rw_h,rw_ca,rw_z,rw_x,ca,x,y,w,h,       o_prob,o_class,av_bin,lamp" << std::endl;
};

void Saver::close() {
    mycsvfile.close();
}

void Saver::save_csv(const Frame &fr) {
    for (auto i = 0; i < fr.basic_params.size(); i++) {
        mycsvfile << *counter << "," << i << "," << fr.basic_params[i].rw_w << "," << fr.basic_params[i].rw_h << "," << 
        fr.basic_params[i].rw_ca << "," << fr.basic_params[i].rw_d << "," << fr.basic_params[i].rw_xc << "," <<   
        fr.basic_params[i].ca << "," << fr.basic_params[i].rect.x << "," << fr.basic_params[i].rect.y << "," << 
        fr.basic_params[i].rect.width << "," << fr.basic_params[i].rect.height << "," 

        << std::endl;
    }
}
