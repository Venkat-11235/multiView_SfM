#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/flann.hpp>
#include<iostream>


namespace featureExtractionMatching{
    struct imgFeatures
    {
        std::string img_name;
        cv::Mat img_gray;
        cv::Mat img_col;
        int desc_rows;
        int desc_cols;
        std::vector<cv::KeyPoint> kp_img;
        cv::Mat desc_img;
 
    };
    imgFeatures readImageFeatures(const cv::Mat& img_grayscale);

    struct imgMatcher
    {
        const float loewe_factor = 0.75f;
        std::vector<cv::DMatch> goodMatches;
        std::vector<cv::Point2f> src_pts;
        std::vector<cv::Point2f> dst_pts;
        std::vector<cv::DMatch> matches_passing_homography;
        std::vector<cv::Point2f> src_gm_corrected;
        std::vector<cv::Point2f> dst_gm_corrected;
        

    };

    imgMatcher featureMatcher(const cv::Mat& desc1, const cv::Mat& desc2, const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2);
    
}
