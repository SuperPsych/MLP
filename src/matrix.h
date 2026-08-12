#pragma once
#include <bits/stdc++.h>
#include <random>
using namespace std;

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
    void times(const vector<double>& x, vector<double>& y) {
        for (int i = 0; i < rows.size(); i++) {
            y[i] = dot(rows[i],x);
        }
    }
    vector<double> times(const vector<double>& x) {
        vector<double> y(rows.size());
        times(x, y);
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
