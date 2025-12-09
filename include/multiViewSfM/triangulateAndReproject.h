#include<opencv2/opencv.hpp>
#include<opencv2/flann.hpp>
#include<iostream>

namespace triangulatePoints{
    struct pointCloudData
    {
        cv::Mat projection_matrix_src;
        cv::Mat projection_matrix_dst;
        cv::Mat point_cloud_data_raw;
        std::vector<cv::Point3f> point_cloud_data_3d;


    };
    pointCloudData triangulate_points(cv::Mat& K1, cv::Mat& K2, cv::Mat& R_0, cv::Mat& t_0, cv::Mat& R_1, cv::Mat& t_1, std::vector<cv::Point2f>& src_pts, std::vector<cv::Point2f>& dst_pts);
}

namespace reprojection{
    struct reprojectionErrorComp{
        float reprojection_error;
        cv::Mat homogeneous_3d_pts;
    };

    reprojectionErrorComp compute_reprojection_error(cv::Mat& obj_pts, std::vector<cv::Point2f>& img_pts, cv::Mat& rot_matrix, cv::Mat& trans_vector, cv::Mat& K, cv::Mat& dist_coeffs);
}
