#pragma once
#include<string>
#include<opencv2/core.hpp>

namespace readIntrinsics{
    struct cameraIntrinsics
    {
        int camera_idx;
        std::string camera_type;
        int width;
        int height;
        cv::Mat K;
        cv::Mat dist;
    };

    cameraIntrinsics readCameraIntrinsics(const std::string& filepath);
    
}