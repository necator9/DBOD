#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <limits>

#include "feature_extractor.hpp"
#include "capturing.hpp"
#include "preprocess.hpp"
#include "config.hpp"


void signal_callback_handler(int signum) {
   std::cout << "Caught signal " << signum << std::endl;
   // Terminate program
   exit(0);
}

int main() {
    signal (SIGINT, signal_callback_handler);
    ConfigParser conf("C:\\Users\\Ivan\\Repositories\\capturing_c\\config.yaml");
    WeightsParser weights("C:\\Users\\Ivan\\Repositories\\capturing_c\\lr_weights.yaml");

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    // std::cout << format(features, cv::Formatter::FMT_NUMPY ) << std::endl << std::endl;

    Capturing cap(CAM_DEV, RESOLUTION.width, RESOLUTION.height, FPS); 
    Preproc prep;
    FeatureExtraxtor fe(FL, CAM_H, IMG_RES, RX_DEG);

    cv::Rect margin_rect = cv::Rect(MARGIN, MARGIN, IMG_RES.width - MARGIN, IMG_RES.height - MARGIN);

    Classifier clf;

    for(auto i = 0; i < 1; i++) {
        Frame fr;
        cap.get_frame(fr.orig_frame);
        prep.prepare_mask(fr, true);
        
        if (fr.basic_params.size() == 0) {
            continue;
        }

        std::vector<BasicObjParams>::iterator f_it = fr.basic_params.end();  // Filtering iterator

        // Filtering by object contour area size if filtering by contour area size is enabled
        if (CA_THR > 0) {
            f_it = std::remove_if(fr.basic_params.begin(), f_it, 
            [](BasicObjParams &bp) { return bp.ca / IMG_RES.area() < CA_THR; });
        }
        // Filtering by intersection with a frame border if filtering is enabled
        if (MARGIN > 0) {
            f_it = std::remove_if(fr.basic_params.begin(), f_it, 
            [&margin_rect](BasicObjParams &bp) { return ((margin_rect & bp.rect).area() < bp.rect.area()); });
        }
        // Filtering by extent coefficient if filtering is enabled
        if (EXTENT_THR > 0) {
            f_it = std::remove_if(fr.basic_params.begin(), f_it, 
            [&margin_rect](BasicObjParams &bp) { return bp.ca / bp.rect.area() < EXTENT_THR; });
        }

        f_it = fr.basic_params.erase(f_it, fr.basic_params.end());
        if (fr.basic_params.size() == 0) {
            continue;
        }

        fe.extract_features(fr);
        
        // Filtering zero distances
        f_it = std::remove_if(fr.basic_params.begin(), f_it, 
        [&margin_rect](BasicObjParams &bp) { return bp.rw_d == 0; });

         // Filtering by distance if filtering is enabled
        if (MAX_DIST > 0) {
            f_it = std::remove_if(fr.basic_params.begin(), f_it, 
            [&margin_rect](BasicObjParams &bp) { return bp.rw_d > MAX_DIST; });
        }

        f_it = fr.basic_params.erase(f_it, fr.basic_params.end());
        if (fr.basic_params.size() == 0) {
            continue;
        }

        cv::Mat_<double> out_probs;
        clf.classify(fr, out_probs, weights);

        std::cout << fr << std::endl;
        std::cout << out_probs << std::endl;
      

    }
    
        

    cap.close();


    // cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    // imshow("Display window", fr.orig_frame);              
    // cv::waitKey(10); 

    return 0;
}

