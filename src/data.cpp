#include <dirent.h>
#include <string>
#include <fstream>
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


//------------------------------Data Methods------------------------------//
Data::Data(void){
    std::string training_data_path = "/Users/jakekasper/Desktop/Mnist/MNIST_Data/trainingSet";
    std::string test_data_path = "/Users/jakekasper/Desktop/Mnist/MNIST_Data/testSet";
//    this->training_files = this->find_training_files(training_data_path);
//    this->test_files = this->find_test_files(test_data_path);
//    this->get_training_set();
//    this->get_test_set();
//    this->generate_binary_data();
//    this->randomize_data();
}

//----------MNIST Data----------//

void Data::MNIST_Data::get_training_data(void){
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

void Data::MNIST_Data::get_test_data(void){
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

void Data::MNIST_Data::randomize_data(void){

}

Eigen::MatrixXd Data::MNIST_Data::process_output(Eigen::MatrixXd input, Eigen::MatrixXd output){
    return output;
}

std::vector< std::vector<std::string> > Data::MNIST_Data::find_training_files(std::string file_path){
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

std::vector<std::string> Data::MNIST_Data::find_test_files(std::string file_path){
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

//----------Binary Data----------//

Data::Binary_Data::Binary_Data(void){
    get_training_data();
}

void Data::Binary_Data::get_training_data(void){
    int example = int(rand() % 65000);
    training_set = get_binary(example);

    solutions = Eigen::MatrixXd::Zero(2, 1);
    if(example % 2 == 0){
        solutions(0, 0) = 1;
        solutions(1, 0) = 0;
    }else{
        solutions(0, 0) = 0;
        solutions(1, 0) = 1;
    }
}

void Data::Binary_Data::get_test_data(void){
    int test = int(rand() % 100) + 65000;
    std::cout << "Test num: " << test << std::endl;
    test_set = get_binary(test);
}

void Data::Binary_Data::randomize_data(void){

}

Eigen::MatrixXd Data::Binary_Data::process_output(Eigen::MatrixXd input, Eigen::MatrixXd output){
    return output;
}

Eigen::VectorXd Data::Binary_Data::get_binary(int num){
    std::bitset<16> map(num);
    std::vector<double> binary;
    Eigen::VectorXd vec(16);
    for(int i = map.size() - 1; i >= 0; --i){
        binary.push_back(double(map[i]));
        vec[15 - i] = double(map[i]);
    }
    return vec;
}

//----------Function Data----------//

Data::Function_Data::Function_Data(void){
    get_training_data();
}

void Data::Function_Data::get_training_data(void){
    //f(x) = x - 5
    int example = int(rand()) % 30 + 5;
    //std::cout << "example: " << example << std::endl;
    training_set = get_binary(example);

    solutions = Eigen::MatrixXd::Zero(1, 1);
    solutions(0, 0) = example - 5;
}

void Data::Function_Data::get_test_data(void){
    //f(x) = x - 5
    int test = int(rand()) % 30 + 5;
    std::cout << "Test num: " << test << std::endl;
    test_set = get_binary(test);
    test_set(0, 0) = test;
}

void Data::Function_Data::randomize_data(void){
}

Eigen::MatrixXd Data::Function_Data::process_output(Eigen::MatrixXd input, Eigen::MatrixXd output){
    int x = input(0, 0);
    Eigen::MatrixXd mod_output = Eigen::MatrixXd::Zero(1, 1);
    for(int i = 0; i < output.rows() - 1; ++i){
        mod_output(0, 0) += output(i, 0) * pow(x, i + 1);
    }
    mod_output(0, 0) += output(output.rows() - 1, 0);
    return mod_output;
}

Eigen::VectorXd Data::Function_Data::get_binary(int num){
    std::bitset<20> map(num);
    std::vector<double> binary;
    Eigen::VectorXd vec(20);
    for(int i = map.size() - 1; i >= 0; --i){
        binary.push_back(double(map[i]));
        vec[19 - i] = double(map[i]);
    }
    return vec;
}

//----------CSV Data----------//
Data::CSV_Data::CSV_Data(void){
    get_training_data();
}

void Data::CSV_Data::get_training_data(void){
    std::string input_filepath = "/Users/jakekasper/Jake_Python/Pytorch/inputs.csv";
    std::string key_filepath = "/Users/jakekasper/Jake_Python/Pytorch/keys.csv";

    std::ifstream inputs;
    std::ifstream keys;

    inputs.open(input_filepath);

    std::string line;
    std::vector< std::vector<std::string> > input_data;
    std::vector<std::string> vec;
    int i = 0;
    while(getline(inputs, line)){
        input_data.push_back(vec);
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ','))
        {
            input_data[i].push_back(cell);
        }
        ++i;
    }

    inputs.close();

    keys.open(key_filepath);

    std::vector<std::string> sols;
    i = 0;
    while(getline(keys, line)){
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ','))
        {
            sols.push_back(cell);
        }
        ++i;
    }

    keys.close();

    Eigen::MatrixXd mat(100, 160);

    for(int i = 0; i < mat.rows(); ++i){
        for(int j = 0; j < mat.cols(); ++j){
            mat(i, j) = stod(input_data[j][i]);
        }
    }

    Eigen::MatrixXd sols_mat(2, 160);

    for(int i = 0; i < sols_mat.cols(); ++i){
        sols_mat(0, i) = stoi(sols[i]);
        if(sols_mat(0, i) == 1){
            sols_mat(1, i) = 0;
        }else{
            sols_mat(1, i) = 1;
        }
    }

    int random_index = rand() % mat.cols();
    training_set = mat.col(random_index);
    solutions = sols_mat.col(random_index);
}

void Data::CSV_Data::get_test_data(void){

}

void Data::CSV_Data::randomize_data(void){

}

Eigen::MatrixXd Data::CSV_Data::process_output(Eigen::MatrixXd input, Eigen::MatrixXd output){
    return output;
}
