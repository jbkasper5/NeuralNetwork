#include <eigen3/Eigen/Core>
#include <vector>
class Topology{
//return the vector of matrices for weights, biases, and activations
//contain and correctly assign the activation functions to each layer -- DONE
//communicate with the data class to make sure that the number of input neurons matches the size of the data vector
public:

    std::vector<int> topology;

    Topology(void);

    void get_topology(void);

    class ActivationFunctions{
    public:
        virtual Eigen::MatrixXd activationFunction(Eigen::MatrixXd m) = 0;
        virtual Eigen::MatrixXd activationFunctionDerivative(Eigen::MatrixXd m) = 0;
    private:
    };

    class Sigmoid: public ActivationFunctions{
    public:
        Eigen::MatrixXd activationFunction(Eigen::MatrixXd m);
        Eigen::MatrixXd activationFunctionDerivative(Eigen::MatrixXd m);
    private:
    };

    class Relu: public ActivationFunctions{
    public:
        Eigen::MatrixXd activationFunction(Eigen::MatrixXd m);
        Eigen::MatrixXd activationFunctionDerivative(Eigen::MatrixXd m);
    private:
    };

    class Tanh: public ActivationFunctions{
    public:
        Eigen::MatrixXd activationFunction(Eigen::MatrixXd m);
        Eigen::MatrixXd activationFunctionDerivative(Eigen::MatrixXd m);
    private:
    };

    class LeakyRelu: public ActivationFunctions{
    public:
        Eigen::MatrixXd activationFunction(Eigen::MatrixXd m);
        Eigen::MatrixXd activationFunctionDerivative(Eigen::MatrixXd m);
    private:
    };

    class LossFunctions{
    public:
        virtual Eigen::MatrixXd lossFunction(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples) = 0;
        virtual Eigen::MatrixXd lossFunctionDerivative(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples) = 0;
    private:
    };

    class LogisticRegression: public LossFunctions{
        Eigen::MatrixXd lossFunction(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples);
        Eigen::MatrixXd lossFunctionDerivative(Eigen::MatrixXd m, Eigen::MatrixXd sols, int numExamples);
    };

private:
};
