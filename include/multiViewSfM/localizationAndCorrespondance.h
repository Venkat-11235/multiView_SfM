#include<iostream>
#include "opencv2/opencv.hpp"


namespace get3D2DCorrespondance{
    struct correspondance
    {
        std::vector<cv::Point3d> filtered3Dpts;
        std::vector<cv::Point2f> filtered2Dpts;
    };
    correspondance get_correspondace(std::vector<cv::DMatch>& filtered_matches, std::vector<cv::KeyPoint>& kp_current, std::vector<cv::DMatch>& prev_2D_filtered_matches, std::vector<cv::Point3d>& prev_3D_points);
}

namespace computeTransformation{
    struct resultTransformation
    {
        cv::Mat rotation_matrix;
        cv::Mat translation_vector;
        std::vector<cv::Point2f> img_point;
        std::vector<cv::Point2f> obj_point;

    };

    resultTransformation computePnP(std::vector<cv::Point3d>& obj_points, std::vector<cv::Point2f>& img_points, cv::Mat& K, cv::Mat& dist_coeffs);
}