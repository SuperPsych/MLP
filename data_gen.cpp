#include <fstream>
#include <random>
using namespace std;

int main() {
    ofstream file("data.txt");

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int i = 0; i < 10000; i++) {
        double x = dist(gen);
        double y = dist(gen);
        double z = dist(gen);
        double w = dist(gen);

        double target = x * y + z * w;

        file << x << " "
             << y << " "
             << z << " "
             << w << " "
             << target << "\n";
    }
}