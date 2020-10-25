#include "topology.h"
#include "data.h"
#include <vector>
#include <eigen3/Eigen/Core>

class Network{
public:
    Network(void);
    void forwardProp(void);
    void backwardProp(void);
    void update_parameters(void);
    void update_data(void);
    void run_test(void);
    int numEpochs;
    int numLayers;
    int numExamples;
    double learning_rate;
    Topology::LossFunctions* func;
    std::vector<Topology::ActivationFunctions*> activationTopology;
    Topology topology;
    Data data;

    Eigen::MatrixXd test_set;

    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::MatrixXd> biases;
    std::vector<Eigen::MatrixXd> activations;
    std::vector<Eigen::MatrixXd> sums;
    std::vector<Eigen::MatrixXd> dw;
    std::vector<Eigen::MatrixXd> db;
    std::vector<Eigen::MatrixXd> dz;
    std::vector<Eigen::MatrixXd> da;
    std::vector<Eigen::MatrixXd> da_prev;
    Eigen::MatrixXd sols;
private:
};
