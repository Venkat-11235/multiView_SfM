#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
#include<iostream>
#include "multiViewSfM/triangulateAndReproject.h"


namespace triangulatePoints{
    pointCloudData triangulate_points(cv::Mat& K1, cv::Mat& K2, cv::Mat& R_0, cv::Mat& t_0, cv::Mat& R_1, cv::Mat& t_1, std::vector<cv::Point2f>& src_pts, std::vector<cv::Point2f>& dst_pts){

        cv::Mat projection_mat_1_temp;
        cv::hconcat(R_0, t_0, projection_mat_1_temp);

        cv::Mat projection_mat_2_temp;
        cv::hconcat(R_1, t_1, projection_mat_2_temp);
        
        triangulatePoints::pointCloudData traingulatedPointCloud;
        traingulatedPointCloud.projection_matrix_src = K1 * projection_mat_1_temp;
        traingulatedPointCloud.projection_matrix_dst = K2 * projection_mat_2_temp;
        
        cv::Mat pts1(2, src_pts.size(), CV_64F);
        cv::Mat pts2(2, dst_pts.size(), CV_64F);

        for (int i = 0; i < src_pts.size(); i++) {
            pts1.at<double>(0, i) = src_pts[i].x;
            pts1.at<double>(1, i) = src_pts[i].y;

            pts2.at<double>(0, i) = dst_pts[i].x;
            pts2.at<double>(1, i) = dst_pts[i].y;
        }
        
        cv::triangulatePoints(traingulatedPointCloud.projection_matrix_src,
                                traingulatedPointCloud.projection_matrix_dst,
                                    pts1,
                                        pts2,traingulatedPointCloud.point_cloud_data_raw );

        traingulatedPointCloud.point_cloud_data_3d.reserve(traingulatedPointCloud.point_cloud_data_raw.cols);         
        
        cv::Mat w = traingulatedPointCloud.point_cloud_data_raw.row(3);                             

        for (int i = 0; i < traingulatedPointCloud.point_cloud_data_raw.cols; i++)
        {
            double w = traingulatedPointCloud.point_cloud_data_raw.at<double>(3, i);

            double X = traingulatedPointCloud.point_cloud_data_raw.at<double>(0, i) / w;
            double Y = traingulatedPointCloud.point_cloud_data_raw.at<double>(1, i) / w;
            double Z = traingulatedPointCloud.point_cloud_data_raw.at<double>(2, i) / w;

            traingulatedPointCloud.point_cloud_data_3d.emplace_back(X, Y, Z);
        }

        return traingulatedPointCloud;
        
    }

}

namespace reprojection{
    reprojectionErrorComp compute_reprojection_error(cv::Mat& obj_pts, std::vector<cv::Point2f>& img_pts, cv::Mat& rot_matrix, cv::Mat& trans_vector, cv::Mat& K, cv::Mat& dist_coeffs){

        reprojection::reprojectionErrorComp reprojection_error_computed;
        
        cv::convertPointsFromHomogeneous(obj_pts.t(), reprojection_error_computed.homogeneous_3d_pts);

        reprojection_error_computed.homogeneous_3d_pts = reprojection_error_computed.homogeneous_3d_pts.reshape(1, reprojection_error_computed.homogeneous_3d_pts.rows);
        reprojection_error_computed.homogeneous_3d_pts.convertTo(reprojection_error_computed.homogeneous_3d_pts, CV_64F);

        cv::Mat rot_vector;
        cv::Rodrigues(rot_matrix, rot_vector);

        std::vector<cv::Point2d> img_pts_from_3d;


        cv::projectPoints(reprojection_error_computed.homogeneous_3d_pts, rot_matrix, trans_vector, K, dist_coeffs, img_pts_from_3d);

        std::vector<cv::Point2d> pts_d;

        for (const auto& p : img_pts) {
            pts_d.emplace_back(static_cast<double>(p.x),
                            static_cast<double>(p.y));
        }

        double total_error = cv::norm(img_pts_from_3d, pts_d,  cv::NORM_L2);

        reprojection_error_computed.reprojection_error = total_error/img_pts.size();

        return reprojection_error_computed;

    } 
}