#include <opencv2/opencv.hpp>
#include <math.h>
#include <limits>
#include <numeric>
#include "feature_extractor.hpp"
#include "config.hpp"



std::ostream& operator<<(std::ostream& os, const Frame& fr) {
    os <<  std::string(55, '_') << std::endl;
    if (fr.basic_params.size() > 0) {
        os << "N obj: " << fr.basic_params.size() << std::endl;
        for (const BasicObjParams& i : fr.basic_params) {
            std::cout << "d: " << i.rw_d << ", h: " << i.rw_h << ", w: " << i.rw_w << 
            ", ca: " << i.rw_ca << std::endl;
        }
    }
    else {
        os << "Empty";
    }
    
    return os;
}

FeatureExtraxtor::FeatureExtraxtor(double fl_, double cam_h_, cv::Size_<int> img_res_, 
double rx_deg_, cv::Mat intrinsic_): fl(fl_), cam_h(cam_h_), img_res(img_res_), rx_deg(rx_deg_), 
intrinsic(intrinsic_) {
    init();
};

FeatureExtraxtor::FeatureExtraxtor(const ConfigParser& conf) {
    fl = conf.focal_length;
    cam_h = conf.height;
    img_res = conf.resolution;
    rx_deg = conf.angle;
    intrinsic = conf.camera_matrix;
    init();
};

void FeatureExtraxtor::init() {
    intrinsic = 
    sens_dim.width = fl * img_res.width / intrinsic.at<double>(0, 0);     // / fx
    sens_dim.height = fl * img_res.height / intrinsic.at<double>(1, 1);   // / fy
    cx_cy = {intrinsic.at<double>(0, 2), intrinsic.at<double>(1, 2)};    
    px_h_mm = sens_dim.height / (fl * img_res.height);
    rx_rad = rx_deg * (M_PI / 180);
       // Rotation matrix around the X axis
    rot_x_mtx = (cv::Mat_<double>(4, 4) <<
        1,          0,           0, 0,
        0, cos(rx_rad), -sin(rx_rad), 0,
        0, sin(rx_rad),  cos(rx_rad), 0,
        0,          0,           0, 1);
    rot_x_mtx_inv = rot_x_mtx.inv();
}

void FeatureExtraxtor::extract_features(Frame &fr) {
    n_obj = fr.basic_params.size();;
    cv::Mat features = cv::Mat_<double>(n_obj, 4);

    cv::Mat boundRect_arr = cv::Mat_<double>(n_obj, 6);
    cv::Mat ca_px = cv::Mat_<double>(n_obj, 1);
    compose_mtx(fr, boundRect_arr, ca_px);

    // Important! Reverse the y coordinates of bound. rect. along y axis before transformations
    cv::Mat px_y_bottom_top = cv::Mat_<double>(n_obj, 2);
    px_y_bottom_top.col(0) = img_res.height - boundRect_arr.col(3);
    px_y_bottom_top.col(1) = img_res.height - boundRect_arr.col(1);

    cv::Mat y_bottom_top_to_hor = cv::Mat_<double>(n_obj, 2);
    y_bottom_top_to_hor.col(0) = (cx_cy.y - px_y_bottom_top.col(0)) * px_h_mm;   
    y_bottom_top_to_hor.col(1) = (cx_cy.y - px_y_bottom_top.col(1)) * px_h_mm;

    // Find atan elementwise
    int cols = y_bottom_top_to_hor.cols, rows = y_bottom_top_to_hor.rows;
    if(y_bottom_top_to_hor.isContinuous()) {
        cols *= rows;
        rows = 1;
    }
    for(int i = 0; i < rows; i++) {
        double* Mi = y_bottom_top_to_hor.ptr<double>(i);
        for(int j = 0; j < cols; j++)
            Mi[j] = atan(Mi[j]);
    }

    cv::Mat rw_distance = features.col(0);
    cv::Mat ang_y_bot_to_hor = y_bottom_top_to_hor.col(0); // Angles to bottom vertices
    estimate_distance(rw_distance, ang_y_bot_to_hor);      // Find object distance in real world

    // Find object height in real world
    cv::Mat rw_height = features.col(1);
    estimate_height(rw_height, rw_distance, y_bottom_top_to_hor);

    // Transform bounding rectangles to a required shape
    cv::Mat px_x_l = cv::Mat_<double>::ones(n_obj, 3);
    boundRect_arr.col(0).copyTo(px_x_l.col(0));
    px_y_bottom_top.col(0).copyTo(px_x_l.col(1));

    cv::Mat px_x_r = cv::Mat_<double>::ones(n_obj, 3);
    boundRect_arr.col(2).copyTo(px_x_r.col(0));
    px_y_bottom_top.col(0).copyTo(px_x_r.col(1));

    cv::Mat px_x_lr;
    cv::vconcat(px_x_l, px_x_r, px_x_lr);

    cv::Mat rw_coords;
    estimate_3d_coordinates(rw_coords, px_x_lr, rw_distance);

    cv::Mat left_bottom = rw_coords(cv::Range(0, rw_coords.rows / 2), cv::Range::all());
    cv::Mat right_bottom = rw_coords(cv::Range(rw_coords.rows / 2, rw_coords.rows), cv::Range::all());
    
    // Find object width in real world
    cv::Mat rw_width = features.col(2);
    rw_width = cv::abs(left_bottom.col(0) - right_bottom.col(0));

    //  Find contour area in real world
    cv::Mat rw_rect_a = rw_width.mul(rw_height);
    cv::Mat px_rect_a = boundRect_arr.col(4).mul(boundRect_arr.col(5));
    cv::Mat rw_ca = features.col(3);
    rw_ca = ca_px.mul(rw_rect_a / px_rect_a);

    decompose_mtx(fr, features);
}

void FeatureExtraxtor::compose_mtx(Frame &fr, cv::Mat &boundRect_arr, cv::Mat &ca_px) {
    // Compose matrix from coordinates of bounding rectangles for convenience
    for(auto r = 0; r < boundRect_arr.rows; r++) {
        double* ptr_br = boundRect_arr.ptr<double>(r);
        double* ptr_ca = ca_px.ptr<double>(r);
        ptr_ca[0] = fr.basic_params[r].ca;
        ptr_br[0] = fr.basic_params[r].rect.x;
        ptr_br[1] = fr.basic_params[r].rect.y;
        ptr_br[2] = fr.basic_params[r].rect.br().x;
        ptr_br[3] = fr.basic_params[r].rect.br().y;
        ptr_br[4] = fr.basic_params[r].rect.width;
        ptr_br[5] = fr.basic_params[r].rect.height;
    }
}

void FeatureExtraxtor::decompose_mtx(Frame &fr, cv::Mat &features) {
    for(auto r = 0; r < features.rows; r++) {
        double* ptr_fe = features.ptr<double>(r);
        fr.basic_params[r].rw_d = ptr_fe[0];
        fr.basic_params[r].rw_h = ptr_fe[1];
        fr.basic_params[r].rw_w = ptr_fe[2];
        fr.basic_params[r].rw_ca = ptr_fe[3];
    }
}

// Estimate distance to the bottom pixel of a bounding rectangle. Based on assumption that object is aligned with the
// ground surface. Calculation uses angle between vertex and optical center along vertical axis
void FeatureExtraxtor::estimate_distance(cv::Mat &distance, const cv::Mat &ang_y_bot_to_hor) {
    double deg;
    double cam_h_abs = abs(cam_h);
    int rows = ang_y_bot_to_hor.rows;
    for(int i = 0; i < rows; i++) {
        double* di = distance.ptr<double>(i);
        const double* ai = ang_y_bot_to_hor.ptr<double>(i);
        deg = ai[0] - rx_rad;
        di[0] = cam_h_abs / (deg >= 0 ? tan(deg) : inf);
    }
}

// Estimate height of object in real world
void FeatureExtraxtor::estimate_height(cv::Mat &height, const cv::Mat &distance, const cv::Mat &ang_y_bot_top_to_hor) {
    double angle_between_pixels, gamma, beta;
    double cam_h_abs = abs(cam_h);
    int rows = distance.rows;
    for(int i = 0; i < rows; i++) {
        double* hi = height.ptr<double>(i);
        const double* di = distance.ptr<double>(i);
        const double* ai = ang_y_bot_top_to_hor.ptr<double>(i);
        angle_between_pixels = abs(ai[0] - ai[1]);
        gamma = atan(di[0] / abs(cam_h));
        beta = M_PI - angle_between_pixels - gamma;
        hi[0] = hypot(cam_h_abs, di[0]) * sin(angle_between_pixels) / sin(beta);
    }
}

// Estimate coordinates of vertices in real world
void FeatureExtraxtor::estimate_3d_coordinates(cv::Mat &rw_coords, const cv::Mat &px_x_lr, const cv::Mat &rw_distance) {
    // Z cam is a scaling factor which is needed for 3D reconstruction
    cv::Mat z_cam_coords = cv::Mat_<double>(n_obj, 1); 
    z_cam_coords = cam_h * sin(rx_rad) + rw_distance * cos(rx_rad);
    cv::Mat z_cam_coords_2x = cv::Mat_<double>(n_obj * 2, 1);
    cv::vconcat(z_cam_coords, z_cam_coords, z_cam_coords_2x);

    cv::Mat cam_xlr_yb_h = cv::Mat_<double>(n_obj * 2, 3);
    cam_xlr_yb_h.col(0) = px_x_lr.col(0).mul(z_cam_coords_2x.col(0)); 
    cam_xlr_yb_h.col(1) = px_x_lr.col(1).mul(z_cam_coords_2x.col(0)); 
    cam_xlr_yb_h.col(2) = px_x_lr.col(2).mul(z_cam_coords_2x.col(0)); 

    // Transform from image plan to camera coordinate system
    cv::Mat camera_coords = intrinsic_inv * cam_xlr_yb_h.t();
    
    // To homogeneous form
    cv::Mat h_row = cv::Mat_<double>::ones(1, camera_coords.cols); 
    camera_coords.push_back(h_row);  
    
    //Transform from to camera to real world coordinate system
    rw_coords = (rot_x_mtx_inv * camera_coords).t();
}

// Increase polynomial order of features
// https://stackoverflow.com/questions/63409333/polynomial-features-in-c
std::vector<double> Classifier::polynomialFeatures(const std::vector<double>& input, unsigned int degree, bool interaction_only, bool include_bias) {
    std::vector<double> features = input;
    std::vector<double> prev_chunk = input;
    std::vector<size_t> indices( input.size() );
    std::iota( indices.begin(), indices.end(), 0 );

    for ( int d = 1 ; d < degree ; ++d ) {
        // Create a new chunk of features for the degree d:
        std::vector<double> new_chunk;
        // Multiply each component with the products from the previous lower degree:
        for ( size_t i = 0 ; i < input.size() - ( interaction_only ? d : 0 ) ; ++i ) {
            // Store the index where to start multiplying with the current component at the next degree up:
            size_t next_index = new_chunk.size();
            for ( auto coef_it = prev_chunk.begin() + indices[i + ( interaction_only ? 1 : 0 )] ; coef_it != prev_chunk.end() ; ++coef_it ) {
                new_chunk.push_back( input[i]**coef_it );
            }
            indices[i] = next_index;
        }
        // Extend the feature vector with the new chunk of features:
        features.reserve( features.size() + std::distance( new_chunk.begin(), new_chunk.end() ) );
        features.insert( features.end(), new_chunk.begin(), new_chunk.end() );

        prev_chunk = new_chunk;
    }
    if ( include_bias )
        features.insert( features.begin(), 1 );

    return features;
}

Classifier::Classifier(const std::string &weight_path) {
    weights = WeightsParser(weight_path);
}

void Classifier::classify(Frame &fr,  cv::Mat &out_probs) {
    std::vector<std::vector<double>> features_poly;
    for (auto &obj : fr.basic_params) {
        std::vector<double> feat_vec = {obj.rw_w, obj.rw_h, obj.rw_ca};
        features_poly.push_back(polynomialFeatures(feat_vec, 2, false, true));
    }

    auto features_poly_flat = flatten<double>(features_poly);
    auto features_poly_m = cv::Mat_<double>(cv::Size(features_poly[0].size(), features_poly.size()));
    std::memcpy(features_poly_m.data,features_poly_flat.data(), features_poly_flat.size() * sizeof(double));
    cv::Mat probs_raw = features_poly_m * weights.coef.t(); // + weights.intercept.t();

    for (auto r = 0; r < probs_raw.rows; r++) {
        probs_raw.row(r) += weights.intercept;

        // Softmax
        double m_max;
        cv::minMaxLoc(probs_raw.row(r), NULL, &m_max, NULL, NULL);
        probs_raw.row(r) -= m_max;
        cv::exp(probs_raw.row(r), out_probs.row(r));
        out_probs.row(r) = out_probs.row(r) / cv::sum(out_probs.row(r));
    }
}

double Classifier::myproduct (double x, double* y) {
    return x * (*y);
}

// Matrix multiplication
// https://medium.com/@dr.sunhongyu/c-efficient-matrix-multiplication-example-b23a18990f1e
std::vector<std::vector<double>> Classifier::matMul(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &B) {
    std::vector<std::vector<double>> res;
    // return if either matrix is empty
    if (A.empty() || B.empty() || A[0].empty() || B[0].empty())
        return res;
    
    int mA = static_cast<int>(A.size());
    int nA = static_cast<int>(A[0].size());
    int mB = static_cast<int>(B.size());
    int nB = static_cast<int>(B[0].size());

    // nA == mB to do the matrix multiplication, the result dim is <mA, nB>
    if(nA != mB) 
        return res;
    
    // init res with right size
    res.resize(mA, std::vector<double>(nB));
    // get a pointer array for the entire column in B matrix
    std::vector<double*> colPtr;

    for(int i = 0; i < mB; i++) {
        colPtr.push_back(&B[i][0]);
    } 
    
    // loop over output columns first, because column element addresses are not continuous
    for(int c = 0; c < nB; c++) {
        for(int r = 0; r < mA; r++) {
            res[r][c] = inner_product(A[r].begin(), A[r].end(), colPtr.begin(), 0, std::plus<double>(), myproduct);
        }
        // move column pointer array to the next column
        transform(colPtr.begin(), colPtr.end(), colPtr.begin(), [](double* x){return ++x;});
    }

    return res;
}

// Transpose vector of vectors
// https://stackoverflow.com/questions/6009782/how-to-pivot-a-vector-of-vectors
std::vector<std::vector<double>> Classifier::transpose(const std::vector<std::vector<double>> data) {
    // this assumes that all inner vectors have the same size and
    // allocates space for the complete result in advance
    std::vector<std::vector<double>> result(data[0].size(), std::vector<double>(data.size()));
    for (std::vector<double>::size_type i = 0; i < data[0].size(); i++) 
        for (std::vector<double>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& v) {
    std::size_t total_size = 0;
    for (const auto& sub : v)
        total_size += sub.size(); // I wish there was a transform_accumulate
    std::vector<T> result;
    result.reserve(total_size);
    for (const auto& sub : v)
        result.insert(result.end(), sub.begin(), sub.end());
    return result;
}

template std::vector<double> flatten(const std::vector<std::vector<double>>& v);