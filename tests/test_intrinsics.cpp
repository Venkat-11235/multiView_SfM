#include "multiViewSfM/cameraIntrinsics.h"
#include<string>
#include<iostream>

int main(int argc, char** argv){
    const std::string path = argv[1];
    auto intr = readIntrinsics::readCameraIntrinsics(path);
    std::cout<<"Width: "<<intr.width<<std::endl;  
    std::cout<<"Height: "<<intr.height<<std::endl;
    std::cout<<intr.K.cols<<std::endl;
    for(int i=0; i < intr.K.rows; i++){
        for (int j = 0; j < intr.K.cols; j++)
        {
            std::cout<<intr.K.at<double>(i,j)<<"\t";

        }
        std::cout<<"\n";
        
    } 
    return 0;
}