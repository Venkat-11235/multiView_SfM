#include<iostream>
#include<open3d/Open3D.h>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>



using namespace open3d;

int main(int argc, char *argv[]) {
     
     
     std::string filename = argv[2];
     std::ifstream infile(filename);

     if(!infile.is_open()){
        throw std::runtime_error("Failed to open the 3D PC file!!!");
        return -1;
        }

     std::vector<Eigen::Vector3d> points;
     std::vector<Eigen::Vector3d> colors;

     points.reserve(100000);
     colors.reserve(100000);

     std::string line;
     while (std::getline(infile,line))
     {
        if(line.empty()||line[0]=='#'){
            continue;
        }
        std::stringstream ss(line);

        long long point3d_id;
        double x,y,z;
        double r,g,b;
        double error;

        if(!(ss>>point3d_id>>x>>y>>z>>r>>g>>b>>error)){
            continue;
        }

        points.emplace_back(x,y,z);
        colors.emplace_back(r/255.0, g/255.0, b/255.0);


     }
     infile.close();


    auto cloud_ptr = std::make_shared<geometry::PointCloud>();
    
    cloud_ptr->points_ = points;
    //cloud_ptr->colors_ = colors;

    cloud_ptr->NormalizeNormals();
    open3d::visualization::DrawGeometries({cloud_ptr}, "PointCloud", 1600, 900);

    // cloud_ptr->EstimateNormals(
    //     open3d::geometry::KDTreeSearchParamHybrid(0.1, 30));
    // cloud_ptr->NormalizeNormals();


    // int depth = 8;
    // auto result = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(*cloud_ptr, depth);
    // auto mesh = std::get<0>(result);
    // auto densities = std::get<1>(result);

    // double min_density = std::numeric_limits<double>::max();
    // double max_density = std::numeric_limits<double>::lowest();
    // for (double d : densities) {
    //     min_density = std::min(min_density, d);
    //     max_density = std::max(max_density, d);
    // }
    // double thr = min_density + 0.7 * (max_density - min_density);

    // std::vector<bool> remove_mask(mesh->vertices_.size(), false);
    // for (size_t i = 0; i < mesh->vertices_.size(); ++i) {
    //     if (densities[i] < thr) {
    //         remove_mask[i] = true;
    //     }
    // }
    // mesh->RemoveVerticesByMask(remove_mask);

    // open3d::visualization::DrawGeometries({mesh}, "Poisson Mesh");

    return 0;

}


