#include <bits/stdc++.h>
#include <random>
#include <utility>
using namespace std;

class Sample {
public:
    vector<double> x;
    vector<double> y;
    Sample() = default;
    Sample(vector<double> x, vector<double> y) : x(move(x)), y(move(y)) {}
};

vector<Sample> load_samples(
    const string& filename,
    int input_dim,
    int output_dim
) {
    ifstream file(filename);

    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }

    vector<Sample> samples;

    while (true) {
        vector<double> x(input_dim);
        vector<double> y(output_dim);

        for (int i = 0; i < input_dim; i++) {
            if (!(file >> x[i])) {
                if (file.eof()) {
                    return samples;
                }
                throw runtime_error("Malformed input data.");
            }
        }

        for (int i = 0; i < output_dim; i++) {
            if (!(file >> y[i])) {
                throw runtime_error("Incomplete sample at end of file.");
            }
        }

        samples.emplace_back(move(x), move(y));
    }
}

class Matrix {
public:
    vector<vector<double>> rows;
    static void print(const vector<double> &x) {
        for (const double i : x) {
            cout << i << " ";
        }
        cout << endl;
    }

    Matrix(int r, int c) {
        rows.resize(r);
        for (int i = 0; i < r; i++) {
            rows[i].resize(c);
        }
    }

    void randomize() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-1.0, 1.0);

        for (auto& row : rows) {
            for (double& value : row) {
                value = dist(gen);
            }
        }
    }

    static void add(vector<double> &a, const vector<double> &b) {
        for (int i = 0; i < a.size(); i++) {
            a[i] += b[i];
        }
    }
    static void add(Matrix &a, const Matrix &b) {
        for (int i = 0; i < a.rows.size(); i++) {
            add(a.rows[i],b.rows[i]);
        }
    }
    static void clear(vector<double>& a) {
        fill(a.begin(), a.end(), 0.0);
    }
    static void clear(Matrix& a) {
        for (auto& row : a.rows) clear(row);
    }
    vector<double> times(const vector<double>& x) {
        vector<double> y(rows.size());
        for (int i = 0; i < rows.size(); i++) {
            y[i] = dot(rows[i],x);
        }
        return y;
    }
    static double dot(const vector<double> &a, const vector<double> &b) {
        double res = 0;
        for (int i = 0; i < a.size(); i++) {
            res += a[i] * b[i];
        }
        return res;
    }
};

class Transform {
public:
    Matrix A;
    vector<double> b;
    Matrix A_delta;
    vector<double> b_delta;
    int dim_in;
    int dim_out;
    Transform(int dim_in, int dim_out) : A(dim_out, dim_in), b(dim_out), A_delta(dim_out, dim_in), b_delta(dim_out) {
        A.randomize();
        this->dim_in = dim_in;
        this->dim_out = dim_out;
    }
    static void ReLU(vector<double> &x) {
        for (int i = 0; i < x.size(); i++) {
            if (x[i]<0) x[i]=0;
        }
    }
    vector<double> transform(const vector<double>& x) {
        vector<double> y = A.times(x);
        Matrix::add(y,b);
        return y;
    }
    void step(const double lr) {
        for (int r=0; r<A_delta.rows.size(); r++) {
            vector<double>& row = A_delta.rows[r];
            for (double & c : row) {
                c *= -lr;
            }
            b_delta[r] *= -lr;
        }
        Matrix::add(A, A_delta);
        Matrix::add(b, b_delta);
        Matrix::clear(A_delta);
        Matrix::clear(b_delta);
    }
};



class Model {
    vector<Transform> transforms;
    vector<vector<double>> activations;
    public:
    explicit Model(vector<int> sizes) {
        for (int i=0; i<sizes.size()-1; i++) {
            transforms.emplace_back(sizes[i], sizes[i+1]);
        }
    }
    static void activation(vector<double> &x) {
        Transform::ReLU(x);
    }
    static void output_activation(vector<double> &x) {

    }
    vector<double> predict(vector<double> x) {
        for (int i = 0; i < transforms.size()-1; i++) {
            x = transforms[i].transform(x);
            activation(x);
        }
        x = transforms.back().transform(x);
        output_activation(x);
        return x;
    }
    void forward_pass(vector<double> &x) {
        activations.clear();
        activations.emplace_back(x);
        for (int i = 0; i < transforms.size()-1; i++) {
            x = transforms[i].transform(x);
            activation(x);
            activations.emplace_back(x);
        }
        x = transforms.back().transform(x);
        output_activation(x);
        activations.emplace_back(x);
    }
    void backprop(const Sample& sample) {
        vector<double> x = sample.x;
        vector<double> y = sample.y;
        forward_pass(x);
        vector<double> grad(transforms.back().dim_out);

        for (int i=0; i<y.size(); i++) {
            grad[i] = activations.back()[i] - y[i];
        }

        for (int i=0; i<transforms.size(); i++) {
            Transform& transform = transforms[transforms.size()-i-1];
            Matrix& A_delta = transform.A_delta;
            vector<double>& b_delta = transform.b_delta;
            Matrix::add(b_delta, grad);
            vector<double>& activation = activations[activations.size()-i-2];
            for (int r = 0; r < transform.dim_out; r++) {
                for (int c = 0; c < transform.dim_in; c++) {
                    A_delta.rows[r][c] += grad[r]*activation[c];
                }
            }
            vector<double> new_grad(transform.dim_in);
            Matrix& A = transform.A;
            for (int r=0; r<transform.dim_out; r++) {
                vector<double>& row = A.rows[r];
                for (int c=0; c<transform.dim_in; c++) {
                    new_grad[c] += A.rows[r][c] * grad[r];
                }
            }
            grad.swap(new_grad);
            for (int j=0; j<grad.size(); j++) {
                if (activation[j] == 0) grad[j] = 0;
            }
        }
    }
    void train(const vector<Sample>& batch, const double lr) {
        for (int i=0; i<batch.size(); i++) {
            backprop(batch[i]);
        }
        for (Transform& transform : transforms) {
            transform.step(lr/batch.size());
        }
    }
    static double loss(vector<double> a, vector<double> p) {
        double loss = 0.0;
        for (int i = 0; i < a.size(); i++) {
            loss += pow(a[i] - p[i],2);
        }
        return loss/(2.0 * a.size());
    }

};

int main() {
    int input_dim = 4;
    int output_dim = 1;
    int hidden_dim = 10;
    int hidden_num = 2;
    vector<int> sizes;
    sizes.emplace_back(input_dim);
    for (int i=0; i<hidden_num; i++) {
        sizes.emplace_back(hidden_dim);
    }
    sizes.emplace_back(output_dim);
    auto model = Model(sizes);
    auto x = vector<double>({2.5, 2, 4, 1});
    auto a = vector<double>({9});
    vector<Sample> samples = load_samples("data.txt", input_dim, output_dim);
    int epochs = 50;
    int batch_size = 64;
    double lr = 0.0001;
    for (int i = 0; i < epochs; i++) {
        vector<Sample> batch(batch_size);
        for (int b = 0; b < samples.size()/batch_size; b++) {
            for (int j=0; j<batch_size; j++) {
                batch[j] = samples[b*batch_size + j];
            }
            model.train(batch, lr);
        }
        if (i % 1 == 0) {
            cout << i << "/" << epochs;
            auto y = model.predict(x);
            cout << " loss: " << Model::loss(y, a) << endl;
        }
    }
    auto y = model.predict(x);
    Matrix::print(y);
}