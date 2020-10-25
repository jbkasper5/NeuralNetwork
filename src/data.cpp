#include <dirent.h>
#include <string>
#include <iostream>
#include <vector>
#include <iomanip>
#include <eigen3/Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>
#include "data.h"

Data::Data(void){
    std::string training_data_path = "/Users/jakekasper/Desktop/Mnist/MNIST_Data/trainingSet";
    std::string test_data_path = "/Users/jakekasper/Desktop/Mnist/MNIST_Data/testSet";
//    this->training_files = this->find_training_files(training_data_path);
//    this->test_files = this->find_test_files(test_data_path);
//    this->get_training_set();
//    this->get_test_set();
    this->generate_binary_data();
    this->randomize_data();
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
                    ++this->num_training_images;
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

void Data::get_training_set(void){
    std::cout << "Collecting training data..." << std::endl;
    solutions = Eigen::MatrixXd::Zero(10, num_training_images);
    int file_num = 0;
    for(int i = 0; i < training_files.size(); ++i){
        for(int j = 0; j < training_files[i].size(); ++j){
            cv::Mat img = cv::imread(this->training_files[i][j], 0);
            Eigen::MatrixXd image;
            cv2eigen(img, image);
            image.resize(image.cols() * image.rows(), 1);
            if(i == 0 && j == 0){
               this->training_set.resize(image.rows(), num_training_images);
            }
            this->training_set.col(file_num) = (image.array() / 255);
            this->solutions(i, file_num) = 1;
            ++file_num;
        }
    }
    std::cout << "Training data collected." << std::endl;
}

void Data::get_test_set(void){
    int file_num = 0;
    for(int i = 0; i < training_files.size(); ++i){
        for(int j = 0; j < training_files[i].size(); ++j){
            cv::Mat img = cv::imread(this->training_files[i][j], 0);
            Eigen::MatrixXd image;
            cv2eigen(img, image);
            image.resize(image.cols() * image.rows(), 1);
            if(i == 0 && j == 0){
                this->test_set.resize(image.rows(), num_training_images);
                std::cout << img << std::endl;
            }
            this->test_set.col(file_num) = (image.array() / 255);
            ++file_num;
        }
    }
}

void Data::randomize_data(void){
    std::cout << "Randomizing training data..." << std::endl;
    srand(time(0));
    for(int i = 0; i < (.6 * num_training_images); ++i){
        int num1 = rand() % num_training_images;
        int num2 = rand() % num_training_images;

        Eigen::VectorXd setTemp = training_set.col(num1);
        Eigen::VectorXd solTemp = solutions.col(num1);

        training_set.col(num1) = training_set.col(num2);
        solutions.col(num1) = solutions.col(num2);

        training_set.col(num2) = setTemp;
        solutions.col(num2) = solTemp;
    }
    std::cout << "Training data randomized." << std::endl;
}

void Data::generate_binary_data(void){
    int num_examples = 1;
    this->solutions = Eigen::MatrixXd::Zero(2, num_examples);
    this->training_set.resize(16, num_examples);
    for(int i = 0; i < num_examples; ++i){
        int num = rand() % 65536;
        this->training_set.col(i) = get_binary(num);
        if(get_binary(num)[15] == 0){
            this->solutions(0, i) = 1;
        }else{
            this->solutions(1, i) = 1;
        }
    }

    std::cout << this->training_set << std::endl << std::endl;
    std::cout << this->solutions << std::endl << std::endl;
}

Eigen::VectorXd Data::get_binary(int num){
    std::bitset<16> map(num);
    std::vector<double> binary;
    Eigen::VectorXd vec(16);
    for(int i = map.size() - 1; i >= 0; --i){
        binary.push_back(double(map[i]));
        vec[15 - i] = double(map[i]);
    }
    return vec;
}
