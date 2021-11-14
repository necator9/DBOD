#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <math.h>
#include <limits>

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

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(fg_frame, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // Using a for loop with iterator
    for(auto rit = std::rbegin(contours); rit != std::rend(contours); ++rit) {
        std::cout << *rit << "\n";
    std::cout << "\n\n";
}
}

void signal_callback_handler(int signum) {
   std::cout << "Caught signal " << signum << std::endl;
   // Terminate program
   exit(0);
}



// Extract object features from given bounding rectangles and contour areas
class FeatureExtraxtor {
public:
    FeatureExtraxtor(double f_l_, cv::Size_<int> img_res_, double r_x_deg_);
    std::vector<cv::Point> contour1 = {cv::Point(587, 476), cv::Point(584, 479), cv::Point(590, 479)};  
    std::vector<cv::Point> contour2 = {cv::Point(587, 400), cv::Point(584, 400), cv::Point(590, 300)};  
    std::vector<std::vector<cv::Point>> contours = {contour1, contour2}; 

    std::vector<std::vector<cv::Point>> contours_poly;
    std::vector<cv::Rect> boundRect;
    std::vector<double> c_a;

    std::vector<double> intrinsic_v = {602.17434328, 0, 511.32476428,
                                      0.0, 601.27444228, 334.8572872,
                                      0, 0, 1};
    cv::Mat intrinsic = cv::Mat(3, 3, CV_32SC1, intrinsic_v.data());

    double r_x_rad;           // Camera rotation angle about x axis in radians
    double cam_h;         // Ground y coord relative to camera (cam. is origin) in meters
    cv::Size_<int> img_res;    // Image resolution (width, height) in px
    double f_l;                  // Focal length in mm
    cv::Size_<double> sens_dim;   // Camera sensor dimensions (width, height) in mm
    cv::Point_<double> cx_cy;
    double px_h_mm;              // Scaling between pixels in millimeters
    double inf = std::numeric_limits<double>::infinity();
    void find_basic_params();
    template<typename T_d, typename T_ybh>
    void estimate_distance(T_d &distance, const T_ybh &y_bot_hor, int col_id);
};

FeatureExtraxtor::FeatureExtraxtor(double f_l_, cv::Size_<int> img_res_, double r_x_deg_):
f_l(f_l_), img_res(img_res_){
    sens_dim.width = f_l * img_res.width / intrinsic.at<double>(0, 0);   // / fx
    sens_dim.height = f_l * img_res.height / intrinsic.at<double>(1, 1);   // / fy
    cx_cy = {intrinsic.at<double>(0, 2), intrinsic.at<double>(1, 2)};    
    px_h_mm = sens_dim.height / (f_l * img_res.height);
    r_x_rad = r_x_deg_ * (M_PI / 180);
};

void FeatureExtraxtor::find_basic_params(){
    for(size_t i = 0; i < contours.size(); i++)
    {
        //approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect.push_back(boundingRect(contours[i]));
        c_a.push_back(contourArea(contours[i]));
    }

    // Transform bounding rectangles to required shape
    // Important! Reverse the y coordinates of bound.rect. along y axis before transformations (self.img_res[1] - y)
    std::vector<std::vector<double>> px_y_bottom_top, y_bottom_top_to_hor;
    double px_y_bottom_top_p1, px_y_bottom_top_p2;
    for(size_t i = 0; i < contours.size(); i++){
        px_y_bottom_top_p1 = (double)img_res.height - boundRect[i].br().y;
        px_y_bottom_top_p2 = (double)img_res.height - boundRect[i].y;
        px_y_bottom_top.push_back(std::vector<double> {px_y_bottom_top_p1, px_y_bottom_top_p2});
        // Distances from vertices to img center (horizon) along y axis, in px
        y_bottom_top_to_hor.push_back(std::vector<double> {cx_cy.y - px_y_bottom_top_p1, cx_cy.y - px_y_bottom_top_p2});  
        //Convert to mm and find angle between object pixel and central image pixel along y axis
        y_bottom_top_to_hor[i][0] = atan(y_bottom_top_to_hor[i][0] * px_h_mm); 
        y_bottom_top_to_hor[i][1] = atan(y_bottom_top_to_hor[i][1] * px_h_mm); 
    }
    
        std::vector<double> distance(contours.size());
        estimate_distance(distance, y_bottom_top_to_hor, 0);    // Passed arg is angles to bottom vertices
}

// Find object distance in real world
template<typename T_d, typename T_ybh>
void FeatureExtraxtor::estimate_distance(T_d &distance, const T_ybh &y_bot_hor, const int col_id) {
    double deg;
    for (auto i = 0; i < distance.size(); i++) {
        deg = y_bot_hor[i][col_id] - r_x_rad;
        distance[i] = abs(cam_h) / (deg >= 0 ? tan(deg) : inf);
    }

}


template <size_t rows, size_t cols>
void process_2d_array_template(int (&array)[rows][cols])
{
    std::cout << __func__ << std::endl;
    for (size_t i = 0; i < rows; ++i)
    {
        std::cout << i << ": ";
        for (size_t j = 0; j < cols; ++j)
            std::cout << array[i][j] << '\t';
        std::cout << std::endl;
    }
}

int main() {
    double r_x_deg = -20;
    double cam_h = -3;
    cv::Size_<int> img_res  = {1024, 768};
    double f_l = 2.2;

    FeatureExtraxtor fe(f_l, img_res, r_x_deg);
    fe.find_basic_params();


    return 0;
}

