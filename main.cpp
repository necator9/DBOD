#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
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
    double r_x_deg = -20;
    double cam_h = -3;
    cv::Size_<int> img_res  = {1024, 768};
    double f_l = 2.2;

    FeatureExtraxtor fe(f_l, img_res, r_x_deg);
    fe.find_basic_params();


    Capturing cap(CAM_DEV, RESOLUTION.width, RESOLUTION.height, FPS); 
    Preproc prep;
    
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
    cv::Mat orig_frame, fg_frame;
    for(auto i = 0; i < 100; i++) {
        cap.get_frame(orig_frame);
        prep.prepare_mask(orig_frame, fg_frame);
    //    std::cout << i << RESOLUTION.width << std::endl;
        imshow( "Display window", orig_frame);                // Show our image inside it.
        cv::waitKey(10); // Wait for a keystroke in the window
    }
    


    return 0;
}

