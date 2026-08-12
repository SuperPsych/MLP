#include <bits/stdc++.h>
#include <random>
#include <utility>
#include <omp.h>
using namespace std;

struct Sample {
    vector<double> x;
    vector<double> y;
    Sample() = default;
    Sample(vector<double>& x, vector<double>& y) : x(move(x)), y(move(y)) {}
    void print(){
        cout << "x: ";
        for(double i : x) cout << i << ", ";
        cout << "y: ";
        for(double i : y) cout << i << ", ";
    }
};


struct Matrix {
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
        double std_dev = sqrt(2.0 / rows[0].size());
        normal_distribution<double> dist(0.0, std_dev);
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

struct Transform {
    Matrix A;
    vector<double> b;
    int dim_in;
    int dim_out;
    Transform(int dim_in, int dim_out) : A(dim_out, dim_in), b(dim_out){
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
    void step(Matrix& A_delta, vector<double>& b_delta, double lr) {
        for (int r=0; r<A_delta.rows.size(); r++) {
            vector<double>& row = A_delta.rows[r];
            for (double & c : row) {
                c *= -lr;
            }
            b_delta[r] *= -lr;
        }
        Matrix::add(A, A_delta);
        Matrix::add(b, b_delta);
    }
};

struct ThreadData {
    vector<Matrix> A_deltas;
    vector<vector<double>> b_deltas;
    vector<vector<double>> activations;

    ThreadData(const vector<int>& dims) {
        A_deltas.reserve(dims.size()-1);
        b_deltas.reserve(dims.size()-1);
        activations.reserve(dims.size());
        for (int i=0; i<dims.size()-1; i++) {
            A_deltas.emplace_back(dims[i+1], dims[i]);
            b_deltas.emplace_back(dims[i+1]);
            activations.emplace_back(dims[i]);
        }
        activations.emplace_back(dims.back());
    }

    void clear(){
        for(auto& A_delta : A_deltas){
            Matrix::clear(A_delta);
        }
        for(auto& b_delta : b_deltas){
            Matrix::clear(b_delta);
        }
        for(auto& layer : activations){
            Matrix::clear(layer);
        }
    }

};

struct Model {
    vector<Transform> transforms;
    Model(const vector<int>& dims) {
        for (int i=0; i<dims.size()-1; i++) {
            transforms.emplace_back(dims[i], dims[i+1]);
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
    void forward_pass(const vector<double> &x, vector<vector<double>>& activations) {
        activations[0] = x;
        for (int i = 0; i < transforms.size()-1; i++) {
            activations[i+1] = transforms[i].transform(activations[i]);
            activation(activations[i+1]);
        }
        activations.back() = transforms.back().transform(activations[activations.size()-2]);
        output_activation(activations.back());
    }
    void backprop(const Sample& sample, ThreadData& td) {
        const vector<double>& x = sample.x;
        const vector<double>& y = sample.y;
        forward_pass(x, td.activations);
        vector<double> grad(transforms.back().dim_out);

        for (int i=0; i<y.size(); i++) {
            grad[i] = td.activations.back()[i] - y[i];
        }

        for (int i=0; i<transforms.size(); i++) {
            int idx = transforms.size()-i-1;
            Transform& transform = transforms[idx];
            Matrix& A_delta = td.A_deltas[idx];
            vector<double>& b_delta = td.b_deltas[idx];
            vector<double>& activation = td.activations[idx];

            Matrix::add(b_delta, grad);
            
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

    void train(const vector<Sample>& samples, int epochs, int batch_size, double lr) {
        omp_set_dynamic(0);
        int thread_num = omp_get_max_threads();
        vector<ThreadData> thread_data(thread_num, ThreadData(dims()));

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int b = 0; b < samples.size()/batch_size; b++) {
                int start = b * batch_size;
                for(ThreadData& td : thread_data){
                    td.clear();
                }
                #pragma omp parallel for
                for(int i = 0; i < batch_size; i++){
                    backprop(samples[start + i], thread_data[omp_get_thread_num()]);
                }

                #pragma omp parallel for
                for(int layer = 0; layer < transforms.size(); layer++){
                    for(int t = 1; t < thread_num; t++){
                        Matrix::add(thread_data[0].A_deltas[layer], thread_data[t].A_deltas[layer]);
                        Matrix::add(thread_data[0].b_deltas[layer], thread_data[t].b_deltas[layer]);
                    }
                }

                for(int t = 0; t < transforms.size(); t++){
                    transforms[t].step(thread_data[0].A_deltas[t], thread_data[0].b_deltas[t], lr / batch_size);
                }
            }
            if(epoch % 20 == 0){
                cout << "Epoch " << epoch << "/" << epochs << " | Loss: " << loss(samples) << endl;
            }
        }
    }

    static double loss(const vector<double>& a, const vector<double>& p) {
        double res = 0.0;
        for (int i = 0; i < a.size(); i++) {
            res += pow(a[i] - p[i],2);
        }
        return res/(2.0 * a.size());
    }

    double loss(const vector<Sample>& samples){
        double res = 0.0;
        for(const Sample& sample : samples){
            res += loss(predict(sample.x), sample.y)/samples.size();
        }
        return res;
    }

    static Model initialize(int input_dim, int output_dim, int hidden_dim, int hidden_num) {
        vector<int> dims;
        dims.emplace_back(input_dim);
        for (int i=0; i<hidden_num; i++) {
            dims.emplace_back(hidden_dim);
        }
        dims.emplace_back(output_dim);
        return Model(dims);
    }

    vector<int> dims() {
        vector<int> dims;
        for (Transform& t : transforms) {
            dims.emplace_back(t.dim_in);
        }
        dims.emplace_back(transforms.back().dim_out);
        return dims;
    }
};


vector<Sample> load_samples(const string& filename, int input_dim, int output_dim){
    ifstream file(filename);
    vector<Sample> samples;
    while(true){
        vector<double> x(input_dim);
        vector<double> y(output_dim);
        for(int i=0; i<input_dim; i++){
            if(!(file >> x[i])) return samples;
        }
        for(int i=0; i<output_dim; i++){
            if(!(file >> y[i])) return samples;
        }
        samples.emplace_back(x, y);
    }
}


int main() {

    int input_dim = 4;
    int output_dim = 1;
    int hidden_dim = 16;
    int hidden_num = 4;

    auto model = Model::initialize(input_dim, output_dim, hidden_dim, hidden_num);
    auto dims = model.dims();

    auto samples = load_samples("./data.txt", input_dim, output_dim);

    int epochs = 500;
    int batch_size = 64;
    double lr = 0.001;
    model.train(samples, epochs, batch_size, lr);
    cout << model.predict({1,1,1,1})[0];
}