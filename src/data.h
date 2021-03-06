class Data{
public:

    Data(void);

    class Data_Assembly{
    public:
        virtual void get_training_data(void) = 0;
        virtual void get_test_data(void) = 0;
        virtual void randomize_data(void) = 0;
        virtual Eigen::MatrixXd process_output(Eigen::MatrixXd input, Eigen::MatrixXd output) = 0;

        Eigen::MatrixXd training_set;
        Eigen::MatrixXd test_set;
        Eigen::MatrixXd solutions;
    };

    class MNIST_Data: public Data_Assembly{
    public:
        void get_training_data(void);
        void get_test_data(void);
        void randomize_data(void);
        Eigen::MatrixXd process_output(Eigen::MatrixXd input, Eigen::MatrixXd output);
    private:
        std::vector< std::vector<std::string> > training_files;
        std::vector<std::string> test_files;
        int num_training_images;

        std::vector< std::vector<std::string> > find_training_files(std::string file_path);
        std::vector<std::string> find_test_files(std::string file_path);
    };

    class Binary_Data: public Data_Assembly{
    public:
        Binary_Data(void);
        void get_training_data(void);
        void get_test_data(void);
        void randomize_data(void);
        Eigen::MatrixXd process_output(Eigen::MatrixXd input, Eigen::MatrixXd output);
    private:
        Eigen::VectorXd get_binary(int num);
    };

    class Function_Data: public Data_Assembly{
    public:
        Function_Data();
        void get_training_data(void);
        void get_test_data(void);
        void randomize_data(void);
        Eigen::MatrixXd process_output(Eigen::MatrixXd input, Eigen::MatrixXd output);
    private:
        Eigen::VectorXd get_binary(int num);
    };

    //temporary class - handle true csv structures in the future
    class CSV_Data: public Data_Assembly{
    public:
        CSV_Data();
        void get_training_data(void);
        void get_test_data(void);
        void randomize_data(void);
        Eigen::MatrixXd process_output(Eigen::MatrixXd input, Eigen::MatrixXd output);
    private:
    };

    class Memory_Storage{
    public:
    private:
    };

private:
};
