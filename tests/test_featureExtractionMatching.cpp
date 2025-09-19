#include "multiViewSfM/featureExtractionMatching.h"
#include<opencv2/core.hpp>
#include<opencv2/flann.hpp>
#include<fstream>
#include<iostream>
#include<filesystem>


void write_u32(std::ofstream& f, uint32_t v){
    f.put(v & 0xFF);
    f.put((v>>8)&0xFF);
    f.put((v<<16)&0xFF);
    f.put((v>>24)&0xFF);
}


uint32_t read_u32(std::ifstream& f){
    uint32_t v = 0;
    int c;
    c = f.get();

    if (c == EOF) throw std::runtime_error("Unexpected EOF while reading u32 1");
    v |= uint32_t(uint8_t(c));

    c = f.get();
    if (c == EOF) throw std::runtime_error("Unexpected EOF while reading u32 2");
    v |= uint32_t(uint8_t(c)) << 8;

    c = f.get();
    if (c == EOF) throw std::runtime_error("Unexpected EOF while reading u32 3");
    v |= uint32_t(uint8_t(c)) << 16;

    c = f.get();
    if (c == EOF) throw std::runtime_error("Unexpected EOF while reading u32 4" );
    v |= uint32_t(uint8_t(c)) << 24;

    return v;
}

int test_saveImgFeatures(const std::string& folder_path){

    std::ofstream f("test_featuresv1.bin", std::ios::binary);
    std::string extractor = "SIFT";
    
    uint32_t len = extractor.size();

    write_u32(f, extractor.size());
    f.write(extractor.data(), extractor.size());

    std::vector<std::filesystem::path> imgs_list;
    for (auto& e: std::filesystem::directory_iterator(folder_path))
    {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();

        for (char &c : ext)
        {
            c = std::tolower(c);
        }
        

        if (ext != ".jpg"&& ext!=".png"&& ext!=".jpeg"){
            throw std::runtime_error("No supporting file formats found!!!");
        }

        imgs_list.push_back(e.path());
    }

    write_u32(f, static_cast<uint32_t>(imgs_list.size()));
    for (int i = 0; i < imgs_list.size(); i++)
    {
        std::cout<<"Image : "<< imgs_list[i]<<std::endl;
    }
    
     
    for (const auto&p : imgs_list)
    {

        cv::Mat img_read;
        img_read = cv::imread(p.string());

        if (img_read.empty())
        {
            throw std::runtime_error("File not found or unable to open the input file !!!");
        }

        cv::cvtColor(img_read, img_read, cv::COLOR_BGR2GRAY);

        featureExtractionMatching::imgFeatures result_img_features;
        result_img_features = featureExtractionMatching::readImageFeatures(img_read);

        std::cout<<"File Name: " <<p.filename()<<std::endl;

        std::string img_name = p.filename().string();

        write_u32(f, img_name.size());
        f.write(img_name.data(), img_name.size());



        write_u32(f, result_img_features.kp_img.size());
        write_u32(f, result_img_features.desc_img.rows);
        write_u32(f, result_img_features.desc_img.cols);
        write_u32(f, 0);

        cv::Mat desc_cont = result_img_features.desc_img.isContinuous() ? result_img_features.desc_img : result_img_features.desc_img.clone();

        f.write((char*)desc_cont.data, desc_cont.total()*desc_cont.elemSize());

        for (const auto& kp: result_img_features.kp_img)
        {
            f.write((char*)&kp.pt.x, sizeof(float));
            f.write((char*)&kp.pt.y, sizeof(float));
            f.write((char*)&kp.size, sizeof(float));
            f.write((char*)&kp.angle, sizeof(float));
            f.write((char*)&kp.response, sizeof(float));
            f.write((char*)&kp.octave, sizeof(int));
            f.write((char*)&kp.class_id, sizeof(int));

        }
    }
    
    return 0;
}

int test_imgFeatureExtraction(std::string& featuresFilePath){


    std::ifstream f(featuresFilePath, std::ios::binary);
    
    uint32_t featMatNameLen = read_u32(f);
    std::string featureMatcher(featMatNameLen, '\0');
    f.read(&featureMatcher[0], featMatNameLen);
    std::cout<<"Feature Matcher: "<<featureMatcher<<std::endl;

    uint32_t total_img_count = read_u32(f);

    std::cout<<"Total Images Present in the bin: "<< total_img_count<< std::endl;

    static_assert(sizeof(float)==4, "float must be 32-bit");
    static_assert(sizeof(int)==4,   "int must be 32-bit");

    for (uint32_t imgIdx = 0; imgIdx < total_img_count; imgIdx++)
    {
        uint32_t imgLen = read_u32(f);
        std::string img_name(imgLen, '\0');
        f.read(&img_name[0], imgLen);

        uint32_t kpsCount = read_u32(f);
        uint32_t rows = read_u32(f);
        uint32_t cols = read_u32(f);
        uint32_t dtype = read_u32(f);

        cv::Mat desc(rows, cols, CV_32F);
        f.read(reinterpret_cast<char*>(desc.data), rows*cols*sizeof(float));

        std::vector<cv::KeyPoint> kps(kpsCount);

        for (uint32_t i = 0; i < kpsCount; i++)
        {
            auto& kp = kps[i];
            f.read(reinterpret_cast<char*>(&kp.pt.x), sizeof(float));
            f.read(reinterpret_cast<char*>(&kp.pt.y), sizeof(float));
            f.read(reinterpret_cast<char*>(&kp.size), sizeof(float));
            f.read(reinterpret_cast<char*>(&kp.angle), sizeof(float));
            f.read(reinterpret_cast<char*>(&kp.response), sizeof(float));
            f.read(reinterpret_cast<char*>(&kp.octave), sizeof(int));
            f.read(reinterpret_cast<char*>(&kp.class_id), sizeof(int));
            if (!f) throw std::runtime_error("READ ERROR: short read!!!");
        }
        std::cout<<"Image "<<imgIdx<<": "<<img_name
                 <<" kp="<<kps.size()
                 <<" desc="<<desc.rows<<"x"<<desc.cols<<"\n";

    }
    

    return 0;
}



int main(int argc, char** argv){
    std::string features_bin_fp = argv[1];
    int res;
    res = test_imgFeatureExtraction(features_bin_fp);
    return 0;
}