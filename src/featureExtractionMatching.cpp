#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
#include<iostream>
#include "multiViewSfM/featureExtractionMatching.h"

namespace {
        cv::Ptr<cv::SIFT>& sift_singleton() {
        static cv::Ptr<cv::SIFT> inst = cv::SIFT::create();
        return inst;
    }
}

namespace{
    cv::BFMatcher def_matcher(){
        static cv::BFMatcher descMatcher(cv::NORM_L2);
        return descMatcher;
    }
}

namespace featureExtractionMatching{
    imgFeatures readImageFeatures(const cv::Mat& img_grayscale){
        

        if(img_grayscale.empty()){
            throw std::runtime_error("Could not open the image !!!");
        }
        if(img_grayscale.channels() != 1){
            throw std::runtime_error("Image not in grayscale !!!");
        }
        auto& sift = sift_singleton();
        
        featureExtractionMatching::imgFeatures img_features;


        sift->detectAndCompute(img_grayscale,cv::noArray(), img_features.kp_img, img_features.desc_img);

        
        return img_features;
    }

    imgMatcher featureMatcher(const cv::Mat& desc1, const cv::Mat& desc2, const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2){

        if (desc1.empty()){
            throw std::runtime_error("Could not find descriptors of src image");
        }
        if (desc2.empty())
        {
            throw std::runtime_error("Could not find descriptors of dst image");
        }

        cv::BFMatcher matcher = def_matcher();
        
        featureExtractionMatching::imgMatcher matched_features;
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(desc1, desc2, knnMatches, 2);

        for (const auto& knnMatch : knnMatches)
        {
            if (knnMatch.size()==2){
                const cv::DMatch& bestMatch = knnMatch[0];
                const cv::DMatch& secondbest = knnMatch[1];

                if(bestMatch.distance < matched_features.loewe_factor*secondbest.distance){
                    matched_features.goodMatches.push_back(bestMatch);
                }
            }
        }

        for (size_t kp_idx = 0; kp_idx < matched_features.goodMatches.size(); kp_idx++)
        {
            matched_features.src_pts.push_back(kp1[matched_features.goodMatches[kp_idx].queryIdx].pt);
            matched_features.dst_pts.push_back(kp2[matched_features.goodMatches[kp_idx].trainIdx].pt);
        }
        std::cout<<"Total matches passing Loewe: "<< matched_features.src_pts.size()<<std::endl;
        std::vector<uchar> inlierMask;
        cv::Mat H = cv::findHomography( matched_features.src_pts,matched_features.dst_pts, cv::RANSAC, 25.0, inlierMask);
        std::cout<<"Length of Inlier Mask: "<<inlierMask.size()<<std::endl;
        for (size_t inlier_idx = 0; inlier_idx < inlierMask.size(); inlier_idx++)
        {
            if(inlierMask[inlier_idx]){
                matched_features.matches_passing_homography.push_back(matched_features.goodMatches[inlier_idx]);
                matched_features.src_gm_corrected.push_back(kp1[matched_features.goodMatches[inlier_idx].queryIdx].pt);
                matched_features.dst_gm_corrected.push_back(kp2[matched_features.goodMatches[inlier_idx].trainIdx].pt);
            }
        }

        return matched_features;
        
    }

}

