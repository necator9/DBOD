#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <chrono>
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

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    // std::cout << format(features, cv::Formatter::FMT_NUMPY ) << std::endl << std::endl;

    Capturing cap(CAM_DEV, RESOLUTION.width, RESOLUTION.height, FPS); 
    Preproc prep;
    FeatureExtraxtor fe(FL, CAM_H, IMG_RES, RX_DEG);


    for(auto i = 0; i < 100; i++) {
        Frame fr;
        cap.get_frame(fr.orig_frame);
        prep.prepare_mask(fr);
        
        if (fr.n_obj == 0) {
            continue;
        }

        fe.extract_features(fr);

        std::cout << fr.features << std::endl;




        
    }
    
    cap.close();


    // cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    // imshow("Display window", fr.orig_frame);              
    // cv::waitKey(10); 

    return 0;
}

