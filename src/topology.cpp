
#include "topology.h"
#include <eigen3/Eigen/Core>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
//------------------------------Topology-------------------------------//

Topology::Topology(void){
    get_topology();
}

void Topology::get_topology(void){
    std::ifstream file("input.txt");
    int num_nodes;
    std::string activationFunction;
    if(file.is_open()){
        while(file >> num_nodes){
            topology.push_back(num_nodes);
        }
    }

    file.close();
    file.open("input.txt");

    if(file.is_open()){
        while(file >> activationFunction){
            for(int i = 0; i < activationFunction.size(); ++i){
                activationFunction[i] = tolower(activationFunction[i]);
            }
            if(activationFunction == "relu"){
                activationTopology.push_back(new Topology::Relu());
            }else if(activationFunction == "leakyrelu"){
                activationTopology.push_back(new Topology::LeakyRelu());
            }else if(activationFunction == "tanh"){
                activationTopology.push_back(new Topology::Tanh());
            }else if(activationFunction == "sigmoid"){
                activationTopology.push_back(new Topology::Sigmoid());
            }else{
                continue;
            }
        }
    }
}

//----------------------Activation Functions----------------------------//

Eigen::MatrixXd Topology::Sigmoid::activationFunction(Eigen::MatrixXd m){
    return (1 / (1 + exp(-m.array())));
}

Eigen::MatrixXd Topology::Sigmoid::activationFunctionDerivative(Eigen::MatrixXd m){
    return Topology::Sigmoid::activationFunction(m).array() - Topology::Sigmoid::activationFunction(m).array().pow(2);
}

Eigen::MatrixXd Topology::Relu::activationFunction(Eigen::MatrixXd m){
    return m.array().max(0);
}

Eigen::MatrixXd Topology::Relu::activationFunctionDerivative(Eigen::MatrixXd m){
    return (m.array().max(0) / m.array());
}

Eigen::MatrixXd Topology::Tanh::activationFunction(Eigen::MatrixXd m){
    return tanh(m.array());
}

Eigen::MatrixXd Topology::Tanh::activationFunctionDerivative(Eigen::MatrixXd m){
    return 1 - Topology::Tanh::activationFunction(m).array().pow(2);
}

Eigen::MatrixXd Topology::LeakyRelu::activationFunction(Eigen::MatrixXd m){
    return (m.array().max(0.01 * m.array()));
}

Eigen::MatrixXd Topology::LeakyRelu::activationFunctionDerivative(Eigen::MatrixXd m){
    return (m.array().max(0.01 * m.array()) / m.array());
}

//--------------------------Loss Functions-------------------------------//

Eigen::MatrixXd Topology::LogLoss::lossFunction(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples){
    Eigen::MatrixXd temp = log(m.array()) * sols.array() + (1 - sols.array()).array() * log((1 - m.array()).array());
    return (temp.array().rowwise().sum() / -numExamples);
}

Eigen::MatrixXd Topology::LogLoss::lossFunctionDerivative(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples){
    return -((sols.array() / m.array())) + ((1 - sols.array()) / (1 - m.array()));
}

Eigen::MatrixXd Topology::MeanSquaredError::lossFunction(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples){
    Eigen::MatrixXd temp;
    return temp;
}

Eigen::MatrixXd Topology::MeanSquaredError::lossFunctionDerivative(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples){
    return (2 * (sols.array() - m.array()));
}

//--------------------------Learning Rate Decay----------------------------//
