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
    topology = {16, 5, 2};
}
//----------------------Activation Functions----------------------------//

Eigen::MatrixXd Topology::Sigmoid::activationFunction(Eigen::MatrixXd m){
    m = exp(-m.array());
    m = 1 + m.array();
    m = m.array().pow(-1);
    return m;
}

Eigen::MatrixXd Topology::Sigmoid::activationFunctionDerivative(Eigen::MatrixXd m){
    Topology::Sigmoid::activationFunction(m);
    m = m.array() - m.array().pow(2);
    return m;
}

Eigen::MatrixXd Topology::Relu::activationFunction(Eigen::MatrixXd m){
    m = m.array().max(0);
    return m;
}

Eigen::MatrixXd Topology::Relu::activationFunctionDerivative(Eigen::MatrixXd m){
    m = (m.array().max(0) / m.array());
    return m;
}

Eigen::MatrixXd Topology::Tanh::activationFunction(Eigen::MatrixXd m){
    m = tanh(m.array());
    return m;
}

Eigen::MatrixXd Topology::Tanh::activationFunctionDerivative(Eigen::MatrixXd m){
    Topology::Tanh::activationFunction(m);
    m = 1 - m.array().pow(2);
    return m;
}

Eigen::MatrixXd Topology::LeakyRelu::activationFunction(Eigen::MatrixXd m){
    m = (m.array().max(0.01 * m.array()));
    return m;
}

Eigen::MatrixXd Topology::LeakyRelu::activationFunctionDerivative(Eigen::MatrixXd m){
    m = (m.array().max(0.01 * m.array()) / m.array());
    return m;
}

//--------------------------Loss Functions-------------------------------//

Eigen::MatrixXd Topology::LogisticRegression::lossFunction(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples){
    Eigen::MatrixXd temp = log(m.array()) * sols.array() + (1 - sols.array()).array() * log((1 - m.array()).array());
    return (temp.array().rowwise().sum() / -numExamples);
}

Eigen::MatrixXd Topology::LogisticRegression::lossFunctionDerivative(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples){
    Eigen::MatrixXd temp = -((sols.array() / m.array())) + ((1 - sols.array()) / (1 - m.array()));
    return temp;
}
