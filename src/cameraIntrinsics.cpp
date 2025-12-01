#include "multiViewSfM/cameraIntrinsics.h"
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<opencv2/core.hpp>
#include<opencv2/core/utils/filesystem.hpp>

namespace readIntrinsics{
    cameraIntrinsics readCameraIntrinsics(const std::string& filepath, const int SCALE_FACTOR){
        std::ifstream in(filepath);
        if (!in.is_open()){
            throw std::runtime_error("Could not open the camera intrinsics file!!! "+ filepath);

        }
        cameraIntrinsics fintr;
        fintr.K = cv::Mat::eye(3,3,CV_64F);
        
        std::string line;
        while (std::getline(in, line))
        {
            if (line.empty())
                continue;
            if (line[0] == '#')
            {
                continue;
            }
            std::istringstream iss(line);
            double fx, fy, cx, cy;
            iss >> fintr.camera_idx;
            iss >> fintr.camera_type;
            iss >> fintr.width;
            iss >> fintr.height;
            iss >> fx;
            iss >> fy;
            iss >> cx;
            iss >> cy;

            fintr.K.at<double>(0,0) = static_cast<int>(fx/SCALE_FACTOR);
            fintr.K.at<double>(0,1) = 0.0;
            fintr.K.at<double>(0,2) = static_cast<int>(cx/SCALE_FACTOR);
            fintr.K.at<double>(1, 0) = 0.0;
            fintr.K.at<double>(1, 1) = static_cast<int>(fy/SCALE_FACTOR);
            fintr.K.at<double>(1, 2) = static_cast<int>(cy/SCALE_FACTOR);
            fintr.K.at<double>(2, 0) = 0.0;
            fintr.K.at<double>(2, 1) = 0.0;
            fintr.K.at<double>(2, 2) = 1.0;


        }
        return fintr;

    }
}



