#include "topology.h"
#include <vector>
#include <eigen3/Eigen/Core>

class Network{
public:
    Network(void);
    void forwardProp();
    void backwardProp(Network net);
    int numEpochs;
    std::vector<Topology::ActivationFunctions*> activationTopology;
private:
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::MatrixXd> biases;
    std::vector<Eigen::MatrixXd> activations;
    std::vector<Eigen::MatrixXd> dw;
    std::vector<Eigen::MatrixXd> db;
    std::vector<Eigen::MatrixXd> dz;
    std::vector<Eigen::MatrixXd> da_prev;

    int numLayers;
};
