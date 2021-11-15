#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
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

    FeatureExtraxtor fe(FL, CAM_H, IMG_RES, RX_DEG);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (auto i = 0; i < 100000; i++)
        fe.find_basic_params();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    // Capturing cap(CAM_DEV, RESOLUTION.width, RESOLUTION.height, FPS); 
    // Preproc prep;
    
    // cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
    // cv::Mat orig_frame, fg_frame;
    // for(auto i = 0; i < 100; i++) {
    //     cap.get_frame(orig_frame);
    //     prep.prepare_mask(orig_frame, fg_frame);
    // //    std::cout << i << RESOLUTION.width << std::endl;
    //     imshow( "Display window", orig_frame);                // Show our image inside it.
    //     cv::waitKey(10); // Wait for a keystroke in the window
    // }
    


    return 0;
}

