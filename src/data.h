class Data{
//determine what kind of data the user wants (custom vs image vs other)
//process the data into a (n, 1) matrix
//communicate with the topology class to make sure n matches the number of neurons in the input layer
public:

    Data(std::string training_data_path, std::string test_data_path);

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

private:
    std::vector< std::vector<std::string> > find_training_files(std::string file_path);
    std::vector<std::string> find_test_files(std::string file_path);

    std::vector< std::vector<std::string> > training_files;
    std::vector<std::string> test_files;

    void process_image(void);
};
