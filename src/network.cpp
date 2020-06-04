#include "network.h"
#include "data.h"
#include <vector>
#include <eigen3/Eigen/Core>
#include <iostream>
Network::Network(void){
    Topology topology;
    Data data("/Users/jakekasper/Desktop/Mnist/MNIST_Data/trainingSet", "/Users/jakekasper/Desktop/Mnist/MNIST_Data/testSet");
    //topology.getInput();

    //get functions to return information from topology class
    //get functions to return the first layer activations from the data class
    //store the loss function in a topology variable
    //initialize the weight vector using the topology vector
    //initialize the bias vector using topology vector
    //initialize the activation vector using topology vector
    //initialize the activation function matrix
    //pull the number of epochs

   // std::vector<Topology::ActivationFunctions*> v;
}

void Network::forwardProp(){
    for(int i = 0; i < this->numLayers - 1; ++i){
        this->activations[i + 1] = (this->weights[i] * this->activations[i]) + this->biases[i];
        this->activationTopology[i]->activationFunction(this->activations[i + 1]);
    }
}

void Network::backwardProp(Network net){

}

int main(int argc, char *argv[]){
    //m = number of examples in the epoch
    //Z matrix = (rows = nodes[i], cols = m)
    //A matrix = Z matrix size
    //W matrix = (rows = nodes[i + 1], cols = nodes[i])
    //B matrix = (rows = nodes[i], cols = m)
    //A[0] = X
    Network net;
    net.forwardProp();
}
