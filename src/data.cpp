#include <dirent.h>
#include <string>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/face.hpp>
#include "data.h"

Data::Data(std::string training_data_path, std::string test_data_path){
    this->training_files = this->find_training_files(training_data_path);
    this->test_files = this->find_test_files(test_data_path);
}

std::vector< std::vector<std::string> > Data::find_training_files(std::string file_path){
    std::string dirname = file_path;
    DIR *pd_1;
    DIR *pd_2;
    struct dirent *dir;
    struct dirent *dir_2;
    pd_1 = opendir(dirname.c_str());
    std::vector< std::vector<std::string> > files (10);

    while((dir = readdir(pd_1)) != NULL){
        if(strncmp(dir->d_name, ".", 1) == 0 || strncmp(dir->d_name, "..", 2) == 0 || strncmp(dir->d_name, ".DS_Store", 9) == 0){
            continue;
        }else{
            std::string files_dir = dirname + '/' + dir->d_name;
            pd_2 = opendir(files_dir.c_str());

            while((dir_2 = readdir(pd_2)) != NULL){
                if(strncmp(dir_2->d_name, ".", 1) == 0 || strncmp(dir_2->d_name, "..", 2) == 0){
                    continue;
                }else{
                    int index = atoi(dir->d_name);
                    files[index].push_back(dirname + "/" + dir->d_name + "/" + dir_2->d_name);
                }
            }
        }
    }

    return files;
}

std::vector<std::string> Data::find_test_files(std::string file_path){
    std::string dirname = file_path;
    DIR *pd;
    struct dirent *dir;
    pd = opendir(dirname.c_str());
    std::vector<std::string> files;

    while((dir = readdir(pd)) != NULL){
        if(strncmp(dir->d_name, ".", 1) == 0 || strncmp(dir->d_name, "..", 2) == 0 || strncmp(dir->d_name, ".DS_Store", 9) == 0){
            continue;
        }else{
            files.push_back(dirname + "/" + dir->d_name);
        }
    }

    return files;
}

void Data::process_image(void){
//    cv::Mat img;
    cv::Mat img2;
}


int main(){
    return 0;
}
