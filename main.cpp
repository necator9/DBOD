#include <iostream>
#include <opencv2/opencv.hpp>

// Move into config later on
auto CAM_DEV = 0;
auto RESOLUTION = cv::Size(640, 480);
auto FPS = 30;
auto CLAHE_LIMIT = 3;
auto CLAHE_GRID_SZ = cv::Size(8, 8);

auto BS_HISTORY = 100;
auto DET_SCHADOWS = true;
auto VAR_THR = 16; //MOG2 thr

auto M_OP_ITER = 3;
auto DIAL_ITER = 0;

class Capturing {
private:
    cv::VideoCapture cap;
    int cam_id, width, height, fps;
public:
    explicit Capturing(int cam_id_ = CAM_DEV, int width_ = RESOLUTION.width, int height_ = RESOLUTION.height,
            int fps = FPS);
    void init_camera();
    void get_frame(cv::Mat& frame);
};

Capturing::Capturing(int cam_id_, int width_, int height_, int fps_):
cam_id(cam_id_), width(width_), height(height_), fps(fps_){
    init_camera();
};

void Capturing::init_camera() {
    cap.open(cam_id);
    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    if (!cap.isOpened()) std::cerr << "Cap is not opened" << std::endl;
}

void Capturing::get_frame(cv::Mat& frame) {
    cap >> frame;
    if (frame.empty()) std::cerr << "Stream has been interrupted" << std::endl;
}


class Preproc {
private:
    cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2(BS_HISTORY, VAR_THR, DET_SCHADOWS);
    //cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorKNN(BS_HISTORY, 400.0, DET_SCHADOWS);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    cv::Mat f_element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

public:
    Preproc();
    void prepare_mask(cv::Mat& orig_frame, cv::Mat& fg_frame);
};

Preproc::Preproc() {
    clahe->setClipLimit(CLAHE_LIMIT);
    clahe->setTilesGridSize(CLAHE_GRID_SZ);
};

void Preproc::prepare_mask(cv::Mat& orig_frame, cv::Mat& fg_frame) {
    cv::cvtColor(orig_frame, orig_frame, cv::COLOR_BGR2GRAY);
    clahe->apply(orig_frame, orig_frame);
    pBackSub->apply(orig_frame, fg_frame);
    cv::morphologyEx(fg_frame, fg_frame, cv::MORPH_OPEN, f_element,
            cv::Point(-1,-1), M_OP_ITER);
    cv::threshold(fg_frame, fg_frame, 170, 255, cv::THRESH_BINARY);
    if (DIAL_ITER > 0) cv::erode(fg_frame, fg_frame, f_element, cv::Point(-1,-1),  DIAL_ITER);
}


int main() {
    Capturing cap;
    Preproc prep;

    cv::Mat orig_frame, fg_frame;
    for(auto i = 0; i < 1000; i++) {
        cap.get_frame(orig_frame);
        prep.prepare_mask(orig_frame, fg_frame);
        std::cout << i << RESOLUTION.width << std::endl;
        cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
        imshow( "Display window", fg_frame);                // Show our image inside it.
        cv::waitKey(10); // Wait for a keystroke in the window
    }


    /*

    cv::String img_name;
    const cv::String dir_path = "/home/ivan/out_img/";
        img_name = dir_path + std::to_string(i) + ".jpg";
        cv::imwrite(img_name, fg_frame);
        std::cout << img_name << " written" << std::endl;
    }

    cap.release();
     */
    return 0;
}

