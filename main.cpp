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

    cv::Rect margin_rect = cv::Rect(MARGIN, MARGIN, IMG_RES.width - MARGIN, IMG_RES.height - MARGIN);


    for(auto i = 0; i < 100; i++) {
        Frame fr;
        cap.get_frame(fr.orig_frame);
        prep.prepare_mask(fr, true);
        
        if (fr.n_obj == 0) {
            continue;
        }

        std::vector<BasicObjParams>::iterator f_it = fr.basic_params.begin();  // Filtering iterator

        // Filtering by object contour area size if filtering by contour area size is enabled
        if (CA_THR > 0) {
            f_it = std::remove_if(f_it, fr.basic_params.end(), 
            [](BasicObjParams &bp) { return bp.ca / IMG_RES.area() < CA_THR; });
        }

        // fr.basic_params.erase(f_it, fr.basic_params.end());
        std::cout << fr.basic_params.end() - f_it  << std::endl;

        // if (MARGIN > 0) {
        //     f_it = std::remove_if(f_it, fr.basic_params.end(), 
        //     [&margin_rect](BasicObjParams &bp) { return ((margin_rect & bp.br).area() > 0); });
        // }

        // std::cout << fr.basic_params.end() - f_it  << std::endl;


                    //std::cout << fr.basic_params.end() - f_it  << std::endl;
            // std::cout << IMG_RES.area() * CA_THR << std::endl;

         //it - vec.begin()

    //    
    //     [](const myobj & o) { return o.m_bMarkedDelete; }),
    // myList.end());
    //     
    //     basic_params = self.filter_c_ar(basic_params, dec_flag=self.c_ar_thr)
    //     basic_params = basic_params[basic_params[:, 0] / self.img_area_px > self.c_ar_thr]

        fe.extract_features(fr);

        // std::cout << fr.features << std::endl;

    }
    
    cap.close();


    // cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    // imshow("Display window", fr.orig_frame);              
    // cv::waitKey(10); 

    return 0;
}

