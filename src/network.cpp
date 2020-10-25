#include "network.h"
#include <vector>
#include <eigen3/Eigen/Core>
#include <iostream>
Network::Network(void){
    //test_set = data.test_set;

    sols = data.solutions;
    std::cout << sols << std::endl << std::endl;

    activationTopology = {new Topology::Relu(), new Topology::Sigmoid};

    numLayers = topology.topology.size();
    learning_rate = 0.05;
    numExamples = data.training_set.cols();

    numEpochs = 1;

    func = new Topology::LogisticRegression();

    for(int i = 0; i < numLayers - 1; ++i){
        Eigen::MatrixXd weight_layer = Eigen::MatrixXd::Random(topology.topology[i + 1], topology.topology[i]);
        weights.push_back(weight_layer);
        weight_layer = weight_layer.array() * 0;
        dw.push_back(weight_layer);

        Eigen::MatrixXd bias_layer = Eigen::MatrixXd::Random(topology.topology[i + 1], 1);
        biases.push_back(bias_layer);
        db.push_back(bias_layer * 0);
    }

    for(int i = 0; i < numLayers; ++i){
        Eigen::MatrixXd activation_layer = Eigen::MatrixXd::Zero(topology.topology[i], numExamples);
        activations.push_back(activation_layer);
        sums.push_back(activation_layer);

        dz.push_back(activation_layer);
        da.push_back(activation_layer);
        da_prev.push_back(activation_layer);
    }

    activations[0] = data.training_set;
    std::cout << activations[0] << std::endl << std::endl;
}

void Network::forwardProp(void){
    for(int i = 0; i < this->numLayers - 1; ++i){
        this->sums[i + 1] = (this->weights[i] * this->activations[i]);
        this->sums[i + 1].colwise() += this->biases[i].col(0);
        this->activations[i + 1] = this->activationTopology[i]->activationFunction(this->sums[i + 1]);



        std::cout << "w: " << this->weights[i] << std::endl << std::endl;
        std::cout << "b: " << this->biases[i] << std::endl << std::endl;
        std::cout << "s: " << this->sums[i + 1] << std::endl << std::endl;
        std::cout << "a: " << this->activations[i + 1] << std::endl << std::endl;

    }
}

void Network::backwardProp(void){
    for(int i = numLayers - 1; i > 0; --i){

//        if(i == (numLayers - 1)){
//            this->da[i] = this->func->lossFunctionDerivative(this->activations[i], this->sols, this->numExamples);
//        }else{
//            this->da[i] = this->da_prev[i];
//        }
//
//        std::cout << "da: " << this->da[i] << std::endl << std::endl;
//
//        this->dz[i] = this->da[i].array() * this->activationTopology[i - 1]->activationFunctionDerivative(this->sums[i]).array();
//
//        std::cout << "dz: " << this->dz[i] << std::endl << std::endl;
//
//        this->dw[i - 1] = this->dz[i] * this->activations[i - 1].transpose();
//        this->dw[i - 1].array() /= this->numExamples;
//
//        std::cout << "dw: " << this->dw[i - 1] << std::endl << std::endl;
//
//        Eigen::MatrixXd temp = this->dz[i].rowwise().sum();
//        this->db[i - 1] = temp.rowwise().sum() / numExamples;
//
//        std::cout << "db: " << this->db[i - 1] << std::endl << std::endl;
//
//        this->da_prev[i - 1] = this->weights[i - 1].transpose() * this->dz[i];
//
//        std::cout << "da_prev: " << this->da_prev[i - 1] << std::endl << std::endl;
   }

    std::cout << "LOSS: " << this->func->lossFunction(this->activations[2], this->sols, this->numExamples) << std::endl << std::endl;
}

void Network::update_parameters(void){
    for(int i = this->dw.size() - 1; i >= 0; --i){
        this->weights[i] = this->weights[i].array() - (this->learning_rate * this->dw[i].array());
        this->biases[i] = biases[i].array() - (this->learning_rate * this->db[i].array());
    }
}

void Network::run_test(void){
    this->activations[0] = this->test_set;
    this->forwardProp();
    std::cout << activations[2].col(0) << std::endl;
}

void Network::update_data(void){
    this->data.generate_binary_data();
    this->activations[0] = data.training_set;
    this->sols = this->data.solutions;
}

int main(int argc, char *argv[]){
    //m = number of examples in the epoch
    //Z matrix = (rows = nodes[i], cols = m)
    //A matrix = Z matrix size
    //W matrix = (rows = nodes[i + 1], cols = nodes[i])
    //B matrix = (rows = nodes[i], cols = m)
    //A[0] = X
    Network net;
    std::cout << "Training network..." << std::endl;
    for(int i = 0; i < net.numEpochs; ++i){
        std::cout << "Epoch " << i + 1<< std::endl;
        std::cout << "training examples: " << std::endl << net.activations[0] << std::endl << std::endl;
        std::cout << "training solutions: " << std::endl << net.sols << std::endl << std::endl;
        net.forwardProp();
        net.backwardProp();
        //net.update_parameters();
        //net.update_data();
    }
//    net.run_test();
    std::cout << "Training complete." << std::endl;
}
