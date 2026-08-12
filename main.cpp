#include "src/model.h"
#include "src/data_loader.h"

int main() {

    int input_dim = 784;
    int output_dim = 10;
    int hidden_dim = 64;
    int hidden_num = 2;

    auto model = Model::initialize(input_dim, output_dim, hidden_dim, hidden_num);

    auto train_samples = load_samples("./mnist_train.txt", input_dim, output_dim);

    int epochs = 50;
    int batch_size = 64;
    double lr = 0.01;
    model.train(train_samples, epochs, batch_size, lr);

    auto test_samples = load_samples("./mnist_test.txt", input_dim, output_dim);
    model.eval(test_samples);

    Matrix::print(model.predict(test_samples[0].x));
}
