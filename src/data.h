class Data{
//determine what kind of data the user wants (custom vs image vs other)
//process the data into a (n, 1) matrix
//communicate with the topology class to make sure n matches the number of neurons in the input layer
public:

    Data(void);

    void generate_binary_data(void);

    class TestData{
    public:
    private:

    };

    class ActualData{
    public:

    private:

    };

    Eigen::MatrixXd training_set;
    Eigen::MatrixXd test_set;
    Eigen::MatrixXd solutions;
    int num_training_images;

private:
    std::vector< std::vector<std::string> > find_training_files(std::string file_path);
    std::vector<std::string> find_test_files(std::string file_path);

    std::vector< std::vector<std::string> > training_files;
    std::vector<std::string> test_files;

    void get_training_set(void);

    void randomize_data(void);

    void get_test_set(void);

    Eigen::VectorXd get_binary(int num);
};
