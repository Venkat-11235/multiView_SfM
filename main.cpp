#include "multiViewSfM/cameraIntrinsics.h"
#include "multiViewSfM/featureExtractionMatching.h"
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <filesystem>
#include <iostream>

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

std::vector<featureExtractionMatching::imgFeatures> saveImgFeatures(const std::string& folder_path){

    std::ofstream f("featuresv1.bin", std::ios::binary);
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

    std::sort(imgs_list.begin(), imgs_list.end());

    write_u32(f, static_cast<uint32_t>(imgs_list.size()));
    
    std::vector<featureExtractionMatching::imgFeatures> image_keypoint_descriptor;
     
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

        std::string img_name = p.filename().string();

        write_u32(f, img_name.size());
        f.write(img_name.data(), img_name.size());
        
        result_img_features.img_name = img_name;
        result_img_features.img_gray = img_read;


        write_u32(f, result_img_features.kp_img.size());
        write_u32(f, result_img_features.desc_img.rows);
        write_u32(f, result_img_features.desc_img.cols);
        write_u32(f, 0);

        result_img_features.desc_cols = result_img_features.desc_img.cols;
        result_img_features.desc_rows = result_img_features.desc_img.rows;

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
        image_keypoint_descriptor.push_back(result_img_features);
    }
    
    return image_keypoint_descriptor;
}

std::vector<featureExtractionMatching::imgFeatures> imgFeatureExtraction(const std::string& featuresFilePath){


    std::ifstream f(featuresFilePath, std::ios::binary);
    
    uint32_t featMatNameLen = read_u32(f);
    std::string featureMatcher(featMatNameLen, '\0');
    f.read(&featureMatcher[0], featMatNameLen);
    std::cout<<"Feature Matcher: "<<featureMatcher<<std::endl;

    uint32_t total_img_count = read_u32(f);

    std::cout<<"Total Images Present in the bin: "<< total_img_count<< std::endl;

    static_assert(sizeof(float)==4, "float must be 32-bit");
    static_assert(sizeof(int)==4,   "int must be 32-bit");

    std::vector<featureExtractionMatching::imgFeatures> image_keypoint_descriptor;

    for (uint32_t imgIdx = 0; imgIdx < total_img_count; imgIdx++)
    {
        featureExtractionMatching::imgFeatures img_keypoints_idx;
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
        
        
        img_keypoints_idx.img_name = img_name;
        img_keypoints_idx.kp_img = kps;
        img_keypoints_idx.desc_cols = cols;
        img_keypoints_idx.desc_rows = rows;
        img_keypoints_idx.desc_img = desc;

        image_keypoint_descriptor.push_back(img_keypoints_idx);
    }
    

    return image_keypoint_descriptor;
}

int main(int argc, char** argv){

    if (argc<4)
    {
        throw std::runtime_error("Bad arguments.\n  Usage: multiview_sfm [--intrinsics|-i PATH_TO_INSTRINSICS] [--folder_path|-fp PATH_TO_IMAGES]\n OR"
         "multiview_sfm [--intrinsics|-i PATH_TO_INSTRINSICS] [--binary_path|-bp PATH_TO_.bin]");
    }
    readIntrinsics::cameraIntrinsics intrinsic_parameters;
    std::vector<featureExtractionMatching::imgFeatures> img_features_dict;

    for (int i = 1; i < argc; i++)
    {
        const std::string arg_value = argv[i];

        if (arg_value=="--intrinsics"||arg_value=="-i")
        {
            const std::string intrinsic_fp = argv[++i];
            intrinsic_parameters = readIntrinsics::readCameraIntrinsics(intrinsic_fp);
            std::cout<<"Intrinsic Read Success Img Dims: "<<intrinsic_parameters.width<<"x"<<intrinsic_parameters.height<<std::endl;
        }
        else if (arg_value=="--folder_path"||arg_value=="-fp")
        {
            const std::string img_folder_path = argv[++i];
            img_features_dict = saveImgFeatures(img_folder_path);
        }
        else if (arg_value=="--binary_path"||arg_value=="-bp"){
            const std::string features_binary_path = argv[++i];
            img_features_dict = imgFeatureExtraction(features_binary_path);
            
            
        }
        else{
            std::cerr<<"Unknown option: "<<arg_value<<std::endl;
            return 1;
        }
        
    }

    if (img_features_dict.empty())
    {
        throw std::runtime_error("No Images / Features found!!!");
        return 1;
    }


    
    for (int imgIdx = 0; imgIdx < img_features_dict.size()-1; imgIdx++)
    {
        featureExtractionMatching::imgFeatures temp_imgFeature1 = img_features_dict[imgIdx];
        featureExtractionMatching::imgFeatures temp_imgFeature2 = img_features_dict[imgIdx+1];
        
        cv::Mat descriptor1 = temp_imgFeature1.desc_img;
        cv::Mat descriptor2 = temp_imgFeature2.desc_img;

        std::vector<cv::KeyPoint> keypoint1 = temp_imgFeature1.kp_img;
        std::vector<cv::KeyPoint> keypoint2 = temp_imgFeature2.kp_img;

        featureExtractionMatching::imgMatcher img_feature_matching_result = featureExtractionMatching::featureMatcher(descriptor1, descriptor2, keypoint1, keypoint2);

        std::cout<<"Image 1: "<< temp_imgFeature1.img_name<<" Image 2: "<< temp_imgFeature2.img_name<<std::endl;
        std::cout<<"Total Good Matches: " << img_feature_matching_result.goodMatches.size()
                <<" Total Matches Passing Homography: "<< img_feature_matching_result.matches_passing_homography.size()<<std::endl;

        #if DEBUG_MODE
            cv::Mat output_show;
            
            cv::drawMatches(temp_imgFeature1.img_gray,keypoint1, temp_imgFeature2.img_gray, keypoint2, img_feature_matching_result.matches_passing_homography, output_show,cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            
            cv::resize(output_show, output_show, cv::Size(static_cast<int>(temp_imgFeature2.img_gray.size().width*(0.25)), 
                                                    static_cast<int>(temp_imgFeature2.img_gray.size().height*(0.25))), cv::INTER_CUBIC);

            cv::imwrite(temp_imgFeature2.img_name, output_show):    
        #endif;
        

        

        cv::Mat E, mask;

        E = cv::findEssentialMat(img_feature_matching_result.src_gm_corrected, img_feature_matching_result.dst_gm_corrected,
                                     intrinsic_parameters.K,cv::RANSAC,0.99,1.0,mask);
        
        std::vector<cv::DMatch> matches_refiltered;
        std::vector<cv::Point2f> src_pts_refiltered;
        std::vector<cv::Point2f> dst_pts_refiltered;

        matches_refiltered.reserve(img_feature_matching_result.matches_passing_homography.size());
        src_pts_refiltered.reserve(img_feature_matching_result.src_gm_corrected.size());
        dst_pts_refiltered.reserve(img_feature_matching_result.dst_gm_corrected.size());


        for (size_t i = 0; i < img_feature_matching_result.matches_passing_homography.size(); i++)
        {
            if (mask.at<uchar>(0, static_cast<int>(i)))
            {
                matches_refiltered.push_back(img_feature_matching_result.matches_passing_homography[i]);
                src_pts_refiltered.push_back(img_feature_matching_result.src_gm_corrected[i]);
                dst_pts_refiltered.push_back(img_feature_matching_result.dst_gm_corrected[i]);

            }
            
        }
        
        img_feature_matching_result.matches_passing_homography.swap(matches_refiltered);
        img_feature_matching_result.src_gm_corrected.swap(src_pts_refiltered);
        img_feature_matching_result.dst_gm_corrected.swap(dst_pts_refiltered);
        
        
        matches_refiltered.clear();
        src_pts_refiltered.clear();
        dst_pts_refiltered.clear();

        cv::Mat rot_mat, trans_mat;
        cv::Mat maskT;

        int inliers = cv::recoverPose(E, img_feature_matching_result.src_gm_corrected, img_feature_matching_result.dst_gm_corrected, 
                             intrinsic_parameters.K, rot_mat, trans_mat, maskT);

        for (size_t i = 0; i < img_feature_matching_result.matches_passing_homography.size(); i++)
        {
            if (mask.at<uchar>(0, static_cast<int>(i)))
            {
                matches_refiltered.push_back(img_feature_matching_result.matches_passing_homography[i]);
                src_pts_refiltered.push_back(img_feature_matching_result.src_gm_corrected[i]);
                dst_pts_refiltered.push_back(img_feature_matching_result.dst_gm_corrected[i]);

            }
            
        }
        
        img_feature_matching_result.matches_passing_homography.swap(matches_refiltered);
        img_feature_matching_result.src_gm_corrected.swap(src_pts_refiltered);
        img_feature_matching_result.dst_gm_corrected.swap(dst_pts_refiltered);
        
        
        matches_refiltered.clear();
        src_pts_refiltered.clear();
        dst_pts_refiltered.clear();


    }
    
    return 0;
}