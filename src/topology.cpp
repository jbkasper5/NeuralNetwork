#include "topology.h"
#include <eigen3/Eigen/Core>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
//----------------------Activation Functions----------------------------//

void Topology::Sigmoid::activationFunction(Eigen::MatrixXd &m){
    m = exp(-m.array());
    m = 1 + m.array();
    m = m.array().pow(-1);
}

void Topology::Sigmoid::activationFunctionDerivative(Eigen::MatrixXd &m){
    Topology::Sigmoid::activationFunction(m);
    m = m.array() - m.array().pow(2);
}

void Topology::Relu::activationFunction(Eigen::MatrixXd &m){
    m = m.array().max(0);
}

void Topology::Relu::activationFunctionDerivative(Eigen::MatrixXd &m){
    m = (m.array().max(0) / m.array());
}

void Topology::Tanh::activationFunction(Eigen::MatrixXd &m){
    m = tanh(m.array());
}

void Topology::Tanh::activationFunctionDerivative(Eigen::MatrixXd &m){
    Topology::Tanh::activationFunction(m);
    m = 1 - m.array().pow(2);
}

void Topology::LeakyRelu::activationFunction(Eigen::MatrixXd &m){
    m = (m.array().max(0.01 * m.array()));
}

void Topology::LeakyRelu::activationFunctionDerivative(Eigen::MatrixXd &m){
    m = (m.array().max(0.01 * m.array()) / m.array());
}

//--------------------------Loss Functions-------------------------------//

void Topology::LogisticRegression::lossFunction(Eigen::MatrixXd &m, Eigen::MatrixXd &sols){
    
}

void Topology::LogisticRegression::lossFunctionDerivative(Eigen::MatrixXd &m, Eigen::MatrixXd &sols){

}
