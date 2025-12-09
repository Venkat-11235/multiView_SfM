#include "multiViewSfM/cameraIntrinsics.h"
#include "multiViewSfM/featureExtractionMatching.h"
#include "multiViewSfM/triangulateAndReproject.h"
#include "multiViewSfM/localizationAndCorrespondance.h"
#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include<open3d/Open3D.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include<vector>

const int SCALE_FACTOR = 6; 

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

        cv::Mat img_read, img_col;
        img_read = cv::imread(p.string());

        if (img_read.empty())
        {
            throw std::runtime_error("File not found or unable to open the input file !!!");
        }
        //cv::resize(img_read, img_col, cv::Size(static_cast<int>(img_read.cols/SCALE_FACTOR), static_cast<int>(img_read.rows/SCALE_FACTOR)),cv::INTER_CUBIC);
        img_col = img_read.clone();
        cv::cvtColor(img_read, img_read, cv::COLOR_BGR2GRAY);
        cv::resize(img_read, img_read, cv::Size(static_cast<int>(img_read.cols/SCALE_FACTOR), static_cast<int>(img_read.rows/SCALE_FACTOR)),cv::INTER_CUBIC);

        featureExtractionMatching::imgFeatures result_img_features;
        result_img_features = featureExtractionMatching::readImageFeatures(img_read);

        std::string img_name = p.filename().string();

        write_u32(f, img_name.size());
        f.write(img_name.data(), img_name.size());
        
        result_img_features.img_name = img_name;
        result_img_features.img_gray = img_read;
        result_img_features.img_col = img_col;


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


auto print_cam = [](const cv::Mat& R_cw, const cv::Mat& t_cw, int idx){
    cv::Mat C = -R_cw.t()*t_cw;
    std::cout<<"Cam : "<<idx<< "center : "<<C.t()<<std::endl;
};


std::vector<cv::Vec3b> getRGBforKeypoint(const cv::Mat& image, const std::vector<cv::Point2f> keypoints, int SCALE_FACTOR){

    std::vector<cv::Vec3b> rgb_values;
    rgb_values.reserve(keypoints.size());

    for(const auto& kp : keypoints){
        int x = static_cast<int>(kp.x);
        int y = static_cast<int>(kp.y);

        x = static_cast<int>(x * SCALE_FACTOR);
        y = static_cast<int>(y * SCALE_FACTOR);

        if (x>=0 && x<image.cols && y>=0 && y<image.rows)
        {
            cv::Vec3b bgr = image.at<cv::Vec3b>(y,x);
            cv::Vec3b rgb(bgr[2], bgr[1], bgr[0]);
            rgb_values.push_back(rgb);

        }
        else{
            std::cout<<"All zeros!!!!"<<std::endl;
            rgb_values.emplace_back(0,0,0);
        }

        
    }

    return rgb_values;

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
            intrinsic_parameters = readIntrinsics::readCameraIntrinsics(intrinsic_fp, SCALE_FACTOR);
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

    cv::Mat R_mat0 = cv::Mat::eye(3,3,CV_64F);
    cv::Mat t_mat0 = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat R_prev, t_prev;
    std::vector<cv::Point3f> prev_3D_points, full_pointcloud_data;
    std::vector<Eigen::Vector3d> points;
    points.reserve(100000);

    std::vector<Eigen::Vector3d> colours;
    std::vector<cv::DMatch> prev_2D_filtered_matches;
    bool is_initialized = false;

    double minX = 1e9, maxX = -1e9, minY=1e9, maxY=-1e9, minZ=1e9, maxZ=-1e9;

    

    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    
    for (int imgIdx = 0; imgIdx < img_features_dict.size()-1; imgIdx++)
    {
        featureExtractionMatching::imgFeatures temp_imgFeature1 = img_features_dict[imgIdx];
        featureExtractionMatching::imgFeatures temp_imgFeature2 = img_features_dict[imgIdx+1];


        std::cout<<"--------------------------------------------"<<std::endl;
        std::cout<<"Size of the image : "<<temp_imgFeature1.img_gray.rows<<"x"<<temp_imgFeature1.img_gray.cols<<std::endl;
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

            cv::imwrite(temp_imgFeature2.img_name, output_show);
        #endif;
        

        

        cv::Mat E;
        cv::Mat mask;

        E = cv::findEssentialMat(img_feature_matching_result.src_gm_corrected, img_feature_matching_result.dst_gm_corrected, intrinsic_parameters.K,cv::RANSAC,0.99,0.9,mask);
        
        std::vector<cv::DMatch> matches_refiltered;
        std::vector<cv::Point2f> src_pts_refiltered;
        std::vector<cv::Point2f> dst_pts_refiltered;

        matches_refiltered.reserve(img_feature_matching_result.matches_passing_homography.size());
        src_pts_refiltered.reserve(img_feature_matching_result.src_gm_corrected.size());
        dst_pts_refiltered.reserve(img_feature_matching_result.dst_gm_corrected.size());


        cv::Mat rot_mat, trans_mat;
        cv::Mat mask2;

        int inliers = cv::recoverPose(E, img_feature_matching_result.src_gm_corrected, img_feature_matching_result.dst_gm_corrected, 
                             intrinsic_parameters.K, rot_mat, trans_mat, mask2);
        for (size_t i = 0; i < static_cast<int>(mask2.total()); i++)
        {
            if (static_cast<int>(mask2.at<uchar>(i)))
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

        if (!is_initialized)
        {
            // cv::Mat R_mat1 = R_mat0 * rot_mat;
            // cv::Mat t_mat1 = t_mat0 + (R_mat0*trans_mat);

            cv::Mat R_mat1 = rot_mat.clone();
            cv::Mat t_mat1 = trans_mat.clone();

            triangulatePoints::pointCloudData triangulated_pc_data = triangulatePoints::triangulate_points(intrinsic_parameters.K, intrinsic_parameters.K,
                                                                                                                R_mat0, t_mat0, R_mat1, t_mat1, 
                                                                                                                img_feature_matching_result.src_gm_corrected,
                                                                                                                img_feature_matching_result.dst_gm_corrected);
            reprojection::reprojectionErrorComp reprojection_error_computed = reprojection::compute_reprojection_error(triangulated_pc_data.point_cloud_data_raw,
                                                                                                                            img_feature_matching_result.dst_gm_corrected,
                                                                                                                                R_mat1, t_mat1, intrinsic_parameters.K, dist_coeffs);
            
            std::cout<<"Reprojection Error: "<<reprojection_error_computed.reprojection_error<<std::endl;

            
            R_prev = R_mat1;

            t_prev = t_mat1;

            prev_3D_points = triangulated_pc_data.point_cloud_data_3d;
            full_pointcloud_data = triangulated_pc_data.point_cloud_data_3d;
            prev_2D_filtered_matches = img_feature_matching_result.matches_passing_homography;
            
            

            std::vector<cv::Vec3b> rgb_pixels = getRGBforKeypoint(temp_imgFeature2.img_col, img_feature_matching_result.dst_gm_corrected, SCALE_FACTOR);

            colours.reserve(full_pointcloud_data.size());
            for (size_t i = 0; i < full_pointcloud_data.size(); i++)
            {
                const cv::Vec3b& rgb = rgb_pixels[i];
                colours.emplace_back(rgb[0] / 255.0,
                                        rgb[1] / 255.0,
                                            rgb[2]/255.0);
            }
            
            
            
            // for (size_t i = 0; i < triangulated_pc_data.point_cloud_data_3d.size(); i++)
            // { 
            //     auto& p = triangulated_pc_data.point_cloud_data_3d[i];
            //     points.emplace_back(p.x, p.y, p.z);

            // }

            for(auto&p : full_pointcloud_data){
                minX = std::min(minX, (double)p.x);
                maxX = std::max(maxX, (double)p.x);
                minY = std::min(minY, (double)p.y);
                maxY = std::max(maxY, (double)p.y);
                minZ = std::min(minZ, (double)p.z);
                maxZ = std::max(maxZ, (double)p.z);
            }

            std::cout<< "Bounds: X["<<minX<<","<<maxX<<"] "
                        <<"Y["<<minY<<","<<maxY<<"] "
                        <<"Z["<<minZ<<","<<maxZ<<"] "<< std::endl;
            
            print_cam(R_mat0, t_mat0, imgIdx);
        }
        else{
            get3D2DCorrespondance::correspondance filtered_correspondances = get3D2DCorrespondance::get_correspondace(img_feature_matching_result.matches_passing_homography, 
                                                                                                                            keypoint2, prev_2D_filtered_matches, prev_3D_points);
            if (filtered_correspondances.filtered3Dpts.size()<4)
            {
                throw std::runtime_error("PNP Computation failed due to insufficient correspondances.");
            }

            computeTransformation::resultTransformation transformed_parameters = computeTransformation::computePnP(filtered_correspondances.filtered3Dpts,
                                                                                                                    filtered_correspondances.filtered2Dpts,
                                                                                                                    intrinsic_parameters.K,
                                                                                                                    dist_coeffs);
                                                                                                   

            triangulatePoints::pointCloudData triangulated_pc_data = triangulatePoints::triangulate_points(intrinsic_parameters.K, intrinsic_parameters.K,
                                                                                                                R_prev, t_prev, transformed_parameters.rotation_matrix, 
                                                                                                                transformed_parameters.translation_vector, 
                                                                                                                img_feature_matching_result.src_gm_corrected,
                                                                                                                img_feature_matching_result.dst_gm_corrected);

            reprojection::reprojectionErrorComp reprojection_error_computed = reprojection::compute_reprojection_error(triangulated_pc_data.point_cloud_data_raw,
                                                                                                                            img_feature_matching_result.dst_gm_corrected,
                                                                                                                                transformed_parameters.rotation_matrix,
                                                                                                                                 transformed_parameters.translation_vector,
                                                                                                                                 intrinsic_parameters.K, dist_coeffs);
            
            std::cout<<"Reprojection Error: "<<reprojection_error_computed.reprojection_error<<std::endl;


            
            std::vector<cv::Point2f> projected_imgpts;
            cv::projectPoints(triangulated_pc_data.point_cloud_data_3d, transformed_parameters.rotation_matrix, transformed_parameters.translation_vector,
                                            intrinsic_parameters.K, dist_coeffs, projected_imgpts);

            cv::Mat output_img_proj = temp_imgFeature2.img_col.clone();
            cv::resize(output_img_proj, output_img_proj, cv::Size(static_cast<int>(output_img_proj.cols/SCALE_FACTOR), static_cast<int>(output_img_proj.rows/SCALE_FACTOR)),cv::INTER_CUBIC);
            std::string fname = "Projected"+ std::to_string(imgIdx)+".jpg";
            
            for (size_t i = 0; i < projected_imgpts.size(); i++)
            {
                const cv::Point2f& proj_pt = projected_imgpts[i];
                const cv::Point2f& obs_pt = img_feature_matching_result.dst_gm_corrected[i];
                
                int proj_x = static_cast<int>(proj_pt.x);
                int proj_y = static_cast<int>(proj_pt.y);
                int obs_x = static_cast<int>(obs_pt.x);
                int obs_y = static_cast<int>(obs_pt.y);

                cv::circle(output_img_proj, cv::Point(proj_x,proj_y), 3, (0,255,0), -1);
                cv::circle(output_img_proj, cv::Point(obs_x, obs_y), 3, (0,0,255), -1);

                cv::line(output_img_proj, cv::Point(proj_x,proj_y), cv::Point(obs_x,obs_y), cv::Scalar(255,0,0), 1);

            }
            cv::imwrite(fname, output_img_proj);
            

            R_prev = transformed_parameters.rotation_matrix;
            t_prev = transformed_parameters.translation_vector;
            prev_3D_points = triangulated_pc_data.point_cloud_data_3d;
            prev_2D_filtered_matches = img_feature_matching_result.matches_passing_homography;
            full_pointcloud_data.insert(full_pointcloud_data.end(),
                       triangulated_pc_data.point_cloud_data_3d.begin(),
                       triangulated_pc_data.point_cloud_data_3d.end());
            
            for(auto&p : full_pointcloud_data){
                minX = std::min(minX, (double)p.x);
                maxX = std::max(maxX, (double)p.x);
                minY = std::min(minY, (double)p.y);
                maxY = std::max(maxY, (double)p.y);
                minZ = std::min(minZ, (double)p.z);
                maxZ = std::max(maxZ, (double)p.z);
            }

            std::cout<< "Bounds: X["<<minX<<","<<maxX<<"] "
                        <<"Y["<<minY<<","<<maxY<<"] "
                        <<"Z["<<minZ<<","<<maxZ<<"] "<< std::endl;

            std::vector<cv::Vec3b> rgb_pixels = getRGBforKeypoint(temp_imgFeature2.img_col, img_feature_matching_result.dst_gm_corrected, SCALE_FACTOR);

            colours.reserve(full_pointcloud_data.size());
            for (size_t i = 0; i < triangulated_pc_data.point_cloud_data_3d.size() && i<rgb_pixels.size(); i++)
            {
                const cv::Vec3b& rgb = rgb_pixels[i];
                colours.emplace_back(rgb[0] / 255.0,
                                        rgb[1] / 255.0,
                                            rgb[2]/255.0);
            }
            print_cam(R_prev, t_prev, imgIdx);
        }
        
            
        is_initialized = true;   
    }
    for (size_t i = 0; i < full_pointcloud_data.size(); i++)
            {
                auto& p = full_pointcloud_data[i];
                points.emplace_back(p.x, p.y, p.z);

            }
    auto point_cloud_data = std::make_shared<open3d::geometry::PointCloud>();

    point_cloud_data->points_ = points;
    point_cloud_data->colors_ = colours;

    point_cloud_data->NormalizeNormals();
    //point_cloud_data->RemoveStatisticalOutliers(20,2.0);
    open3d::visualization::DrawGeometries({point_cloud_data}, "PointCloud", 1000, 1000);

    return 0;
}