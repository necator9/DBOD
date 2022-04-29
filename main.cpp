#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <limits>
#include <numeric>

#include "feature_extractor.hpp"
#include "capturing.hpp"
#include "preprocess.hpp"
#include "config.hpp"
#include "saver.hpp"

volatile sig_atomic_t interrupted = false;

void signal_callback_handler(int signum) {
   std::cout << "Caught signal " << signum << std::endl;
   interrupted = true;
}


void increment(int &counter, std::vector<int> &time_window, std::chrono::steady_clock::time_point begin) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        int it_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        time_window.push_back(it_duration);
        // Log detection speed periodically 
        if (counter % 200 == 0) {
            auto const count = static_cast<float>(time_window.size());
            int sum = 0;
            for (auto i=0; i < count; i++) {
                sum += time_window[i];
            }
            double it_fps = 1 / ((sum / count) / 1000);
            std::cout << "Average FPS: "<< it_fps << std::endl;
            time_window.clear();
        }
        counter++;
        // std::cout << "counter: "<< counter << std::endl;

}

int main() {
    signal (SIGINT, signal_callback_handler);
    const ConfigParser conf = ConfigParser("C:\\Users\\Ivan\\Repositories\\capturing_c\\config.yaml");
    
    Capturing cap(conf); 
    Preproc prep(conf);
    FeatureExtraxtor fe(conf);

    int margin = conf.margin;
    cv::Size res = conf.resolution;
    cv::Rect margin_rect = cv::Rect(margin, margin, res.width - margin, res.height - margin);

    Classifier clf(conf.weights);
    int counter = 0;
    std::vector<int> time_window;
    


    Saver s(conf.out_dir, conf.save_csv, &counter);

    while (not interrupted) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        Frame fr;
        if (not cap.get_frame(fr.orig_frame))
            break;

        prep.prepare_mask(fr, false);

        if (fr.basic_params.size() == 0) {
            increment(counter, time_window, begin);
            continue;
        }

        std::vector<BasicObjParams>::iterator f_it = fr.basic_params.end();  // Iterator for filtering
        // Filtering by object contour area size if filtering by contour area size is enabled
        if (conf.cont_area_thr > 0) {
            f_it = std::remove_if(fr.basic_params.begin(), f_it, 
            [&conf](BasicObjParams &bp) { return bp.ca / conf.resolution.area() < conf.cont_area_thr; });
        }
        // Filtering by intersection with a frame border if filtering is enabled
        if (conf.margin > 0) {
            f_it = std::remove_if(fr.basic_params.begin(), f_it, 
            [&margin_rect](BasicObjParams &bp) { return ((margin_rect & bp.rect).area() < bp.rect.area()); });
        }
        // Filtering by extent coefficient if filtering is enabled
        if (conf.extent_thr > 0) {
            f_it = std::remove_if(fr.basic_params.begin(), f_it, 
            [&conf](BasicObjParams &bp) { return bp.ca / bp.rect.area() < conf.extent_thr; });
        }

        f_it = fr.basic_params.erase(f_it, fr.basic_params.end());

        if (fr.basic_params.size() == 0) {
            increment(counter, time_window, begin);
            continue;
        }
         
        fe.extract_features(fr);
        
        // Filtering zero distances
        f_it = std::remove_if(fr.basic_params.begin(), f_it, 
        [](BasicObjParams &bp) { return bp.rw_d == 0; });

         // Filtering by distance if filtering is enabled
        if (conf.max_distance > 0) {
            f_it = std::remove_if(fr.basic_params.begin(), f_it, 
            [&conf](BasicObjParams &bp) { return bp.rw_d > conf.max_distance; });
        }

        f_it = fr.basic_params.erase(f_it, fr.basic_params.end());
        if (fr.basic_params.size() == 0) {
            increment(counter, time_window, begin);
            continue;
        }

        clf.classify(fr);
        s.save_csv(fr);
        increment(counter, time_window, begin);

        // std::cout << fr << std::endl;
        // std::cout << out_probs << std::endl;
    }
    
    cap.close();
    s.close(); 
    


    // cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    // imshow("Display window", fr.orig_frame);              
    // cv::waitKey(10); 

    return 0;
}



