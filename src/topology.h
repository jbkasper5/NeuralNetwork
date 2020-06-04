#include <eigen3/Eigen/Core>
#include <vector>
class Topology{
//return the vector of matrices for weights, biases, and activations
//contain and correctly assign the activation functions to each layer -- DONE
//communicate with the data class to make sure that the number of input neurons matches the size of the data vector
public:
    class ActivationFunctions{
    public:
        virtual void activationFunction(Eigen::MatrixXd &m) = 0;
        virtual void activationFunctionDerivative(Eigen::MatrixXd &m) = 0;
    private:
    };

    class Sigmoid: public ActivationFunctions{
    public:
        void activationFunction(Eigen::MatrixXd &m);
        void activationFunctionDerivative(Eigen::MatrixXd &m);
    private:
    };

    class Relu: public ActivationFunctions{
    public:
        void activationFunction(Eigen::MatrixXd &m);
        void activationFunctionDerivative(Eigen::MatrixXd &m);
    private:
    };

    class Tanh: public ActivationFunctions{
    public:
        void activationFunction(Eigen::MatrixXd &m);
        void activationFunctionDerivative(Eigen::MatrixXd &m);
    private:
    };

    class LeakyRelu: public ActivationFunctions{
    public:
        void activationFunction(Eigen::MatrixXd &m);
        void activationFunctionDerivative(Eigen::MatrixXd &m);
    private:
    };

    class LossFunctions{
    public:
        virtual void lossFunction(Eigen::MatrixXd &m, Eigen::MatrixXd &sols) = 0;
        virtual void lossFunctionDerivative(Eigen::MatrixXd &m, Eigen::MatrixXd &sols) = 0;
    private:
    };

    class LogisticRegression: public LossFunctions{
        void lossFunction(Eigen::MatrixXd &m, Eigen::MatrixXd &sols);
        void lossFunctionDerivative(Eigen::MatrixXd &m, Eigen::MatrixXd &sols);
    };

private:
};
