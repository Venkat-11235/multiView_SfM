#include<iostream>
#include "opencv2/opencv.hpp"
#include "multiViewSfM/localizationAndCorrespondance.h"


namespace get3D2DCorrespondance{

    correspondance get_correspondace(std::vector<cv::DMatch>& filtered_matches, std::vector<cv::KeyPoint>& kp_current, std::vector<cv::DMatch>& prev_2D_filtered_matches, std::vector<cv::Point3d>& prev_3D_points){

        std::unordered_map<int, int> trainIdx_to_3d_idx;
        correspondance result_3d_2d;

        for (int idx = 0; idx < prev_2D_filtered_matches.size(); ++idx)
        {
            const auto& m = prev_2D_filtered_matches[idx];
            trainIdx_to_3d_idx[m.trainIdx] = idx;
        }
        std::vector<cv::Point3d> common_3d_pts;
        std::vector<cv::Point2f> corresponding_2d_pts;

        for (const auto& m : filtered_matches)
        {
            int q = m.queryIdx;
            int t = m.trainIdx;

            auto it = trainIdx_to_3d_idx.find(q);
            if (it != trainIdx_to_3d_idx.end())
            {
                int idx3d = it->second;
                common_3d_pts.push_back(prev_3D_points[idx3d]);
                corresponding_2d_pts.push_back(kp_current[t].pt);
            }
        }

        if (common_3d_pts.size() < 4)
        {
            std::cout << "[PnP] Not enough valid 3D–2D correspondences: "
                    << common_3d_pts.size() << std::endl;
            return result_3d_2d;
        }

        result_3d_2d.filtered3Dpts = common_3d_pts;
        result_3d_2d.filtered2Dpts = corresponding_2d_pts;

        return result_3d_2d;

    }

}


namespace computeTransformation{
    resultTransformation computePnP(std::vector<cv::Point3d>& obj_points, std::vector<cv::Point2f>& img_points, cv::Mat& K, cv::Mat& dist_coeffs){
        resultTransformation solvedTransformations;
        std::vector<int> inliers;
        cv::Mat rotation_vector;
        bool result = cv::solvePnPRansac(obj_points, img_points, K, dist_coeffs, rotation_vector, solvedTransformations.translation_vector,false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
        
        cv::Rodrigues(rotation_vector, solvedTransformations.rotation_matrix);

        return solvedTransformations;
    }
}

