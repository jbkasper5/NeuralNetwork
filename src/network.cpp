#include "network.h"
#include <vector>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <fstream>
Network::Network(void){
    //test_set = data.test_set;
    assembler = new Data::Function_Data();

    sols = assembler->solutions;

    //Gather Topology
    func = new Topology::MeanSquaredError();
    activationTopology = topology.activationTopology;
    numLayers = topology.topology.size();


    //Establish hyperparameters
    learning_rate = 0.001;
    numEpochs = 500000;
    numExamples = assembler->training_set.cols();

    for(int i = 0; i < numLayers - 1; ++i){
        Eigen::MatrixXd weight_layer = Eigen::MatrixXd::Random(topology.topology[i + 1], topology.topology[i]);
        weight_layer.array() += 1;
        weights.push_back(weight_layer / 40);
        dw.push_back(weight_layer * 0);

        Eigen::MatrixXd bias_layer = Eigen::MatrixXd::Zero(topology.topology[i + 1], 1);
        biases.push_back(bias_layer);
        db.push_back(bias_layer);
    }

    for(int i = 0; i < numLayers; ++i){
        Eigen::MatrixXd activation_layer = Eigen::MatrixXd::Zero(topology.topology[i], numExamples);
        activations.push_back(activation_layer);
        sums.push_back(activation_layer);

        dz.push_back(activation_layer);
        da.push_back(activation_layer);
    }

    activations[0] = assembler->training_set;
}

void Network::forwardProp(void){
    for(int i = 0; i < numLayers - 1; ++i){
        sums[i + 1] = (weights[i] * activations[i]);
        sums[i + 1].colwise() += biases[i].col(0);
        activations[i + 1] = this->activationTopology[i]->activationFunction(sums[i + 1]);

//        std::cout << "w[" << i << "]: " << std::endl << weights[i] << std::endl << std::endl;
//        std::cout << "b[" << i << "]: " << std::endl << biases[i] << std::endl << std::endl;
//        std::cout << "s[" << i << "]: " << std::endl << sums[i + 1] << std::endl << std::endl;
//        std::cout << "a[" << i + 1 << "]: " << std::endl << activations[i + 1] << std::endl << std::endl;
    }
 //       std::cout << "Output " << std::endl << activations[activations.size() - 1] << std::endl << std::endl;
}

void Network::backwardProp(void){
    //std::cout << "Solution " << std::endl << sols << std::endl << std::endl;
    Eigen::MatrixXd loss = this->func->lossFunctionDerivative(assembler->process_output(activations[0], activations[activations.size() - 1]), sols, numExamples);
//    std::cout << "loss: " << std::endl << loss << std::endl << std::endl;
    db[db.size() - 1] = this->activationTopology[this->activationTopology.size() - 1]->activationFunctionDerivative(sums[sums.size() - 1]).array() * loss.array();
    for(int i = numLayers - 2; i > 0; --i){
        dw[i] = db[i] * (activations[i].transpose());
        db[i - 1] = this->activationTopology[i]->activationFunctionDerivative(sums[i]).array() * (weights[i].transpose() * db[i]).array();
    }
    dw[0] = db[0] * (activations[0].transpose());
}

void Network::update_parameters(void){
    for(int i = dw.size() - 1; i >= 0; --i){
        weights[i] += (learning_rate * dw[i]);
        biases[i] += (learning_rate * db[i]);
    }
}

void Network::update_data(void){
    assembler->get_training_data();
    this->activations[0] = assembler->training_set;
    this->sols = this->assembler->solutions;
}

void Network::run_test(void){
    for(int i = 0; i < 20; ++i){
        assembler->get_test_data();
        activations[0] = assembler->test_set;
        forwardProp();
        std::cout << "Output: " << std::endl << activations[activations.size() - 1].array().round() << std::endl << std::endl;
    }
}

int main(int argc, char *argv[]){
    Network net;
    std::cout << "Training network..." << std::endl;
    for(int i = 1; i <= net.numEpochs; ++i){
        //std::cout << "Epoch " << i << std::endl;
        if(i % (net.numEpochs / 10) == 0){
            std::cout << (double(i) / net.numEpochs) * 100 << "%..." << std::endl;
        }
        net.forwardProp();
        net.backwardProp();
        net.update_parameters();
        net.update_data();
    }
    std::cout << "Training complete." << std::endl;
    std::cout << "Running Tests..." << std::endl;
    net.run_test();
}
