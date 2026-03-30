#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <iostream>

using namespace Eigen;

// ─── Activation Functions ────────────────────────────────────────────────────
struct Activation {
    std::function<MatrixXd(const MatrixXd&)> forward;
    std::function<MatrixXd(const MatrixXd&)> derivative;

    static Activation ReLU() {
        return {
            [](const MatrixXd& x) { return x.cwiseMax(0.0); },
            [](const MatrixXd& x) { return (x.array() > 0.0).cast<double>().matrix(); }
        };
    }

    static Activation Sigmoid() {
        return {
            [](const MatrixXd& x) {
                return (1.0 / (1.0 + (-x.array()).exp())).matrix();
            },
            [](const MatrixXd& x) {
                MatrixXd s = (1.0 / (1.0 + (-x.array()).exp())).matrix();
                return (s.array() * (1.0 - s.array())).matrix();
            }
        };
    }

    static Activation Linear() {
        return {
            [](const MatrixXd& x) { return x; },
            [](const MatrixXd& x) { return MatrixXd::Ones(x.rows(), x.cols()); }
        };
    }
};

// ─── Loss Functions ───────────────────────────────────────────────────────────
struct Loss {
    std::function<double(const MatrixXd&, const MatrixXd&)>      compute;
    std::function<MatrixXd(const MatrixXd&, const MatrixXd&)>    gradient;

    static Loss MSE() {
        return {
            [](const MatrixXd& pred, const MatrixXd& target) {
                return (pred - target).squaredNorm() / pred.cols();
            },
            [](const MatrixXd& pred, const MatrixXd& target) {
                return 2.0 * (pred - target) / pred.cols();
            }
        };
    }

    static Loss BinaryCrossEntropy() {
        return {
            [](const MatrixXd& pred, const MatrixXd& target) {
                double eps = 1e-9;
                auto p = pred.array().cwiseMax(eps).cwiseMin(1 - eps);
                return -(target.array() * p.log() + (1 - target.array()) * (1 - p).log()).mean();
            },
            [](const MatrixXd& pred, const MatrixXd& target) {
                double eps = 1e-9;
                auto p = pred.array().cwiseMax(eps).cwiseMin(1 - eps);
                return ((p - target.array()) / (p * (1 - p) + eps)).matrix() / pred.cols();
            }
        };
    }
};

// ─── Layer ────────────────────────────────────────────────────────────────────
struct Layer {
    MatrixXd W;   // weights [out x in]
    VectorXd b;   // biases  [out]
    MatrixXd dW, db_mat;
    Activation act;

    // Cache for backprop
    MatrixXd z_cache;   // pre-activation
    MatrixXd a_cache;   // post-activation (input to next layer)

    Layer(int in_size, int out_size, Activation activation, std::mt19937& rng)
        : act(std::move(activation))
    {
        // He initialization for ReLU, Xavier for others
        double std = std::sqrt(2.0 / in_size);
        std::normal_distribution<double> dist(0.0, std);

        W = MatrixXd(out_size, in_size).unaryExpr([&](double) { return dist(rng); });
        b = VectorXd::Zero(out_size);
        dW = MatrixXd::Zero(out_size, in_size);
        db_mat = MatrixXd::Zero(out_size, 1);
    }
};

// ─── Neural Network ───────────────────────────────────────────────────────────
class NeuralNetwork {
public:
    NeuralNetwork(double learning_rate = 0.01, Loss loss = Loss::MSE(), unsigned seed = 42)
        : lr_(learning_rate), loss_(std::move(loss)), rng_(seed) {}

    // Add a layer: specify output size and activation
    void addLayer(int in_size, int out_size, Activation activation) {
        layers_.emplace_back(in_size, out_size, std::move(activation), rng_);
    }

    // Forward pass — input: [features x batch]
    MatrixXd forward(const MatrixXd& X) {
        MatrixXd a = X;
        for (auto& layer : layers_) {
            layer.a_cache = a;
            // z = W*a + b (broadcast bias)
            MatrixXd z = (layer.W * a).colwise() + layer.b;
            layer.z_cache = z;
            a = layer.act.forward(z);
        }
        return a;
    }

    // Backward pass
    void backward(const MatrixXd& pred, const MatrixXd& target) {
        MatrixXd delta = loss_.gradient(pred, target);

        for (int i = (int)layers_.size() - 1; i >= 0; --i) {
            auto& layer = layers_[i];
            // delta through activation derivative
            MatrixXd dz = delta.cwiseProduct(layer.act.derivative(layer.z_cache));

            int batch = dz.cols();
            layer.dW  = dz * layer.a_cache.transpose() / batch;
            layer.db_mat = dz.rowwise().mean();

            // propagate delta to previous layer
            delta = layer.W.transpose() * dz;
        }
    }

    // Update weights (SGD)
    void updateWeights() {
        for (auto& layer : layers_) {
            layer.W -= lr_ * layer.dW;
            layer.b -= lr_ * layer.db_mat;
        }
    }

    // Train one epoch
    double trainEpoch(const MatrixXd& X, const MatrixXd& Y) {
        MatrixXd pred = forward(X);
        double   loss = loss_.compute(pred, Y);
        backward(pred, Y);
        updateWeights();
        return loss;
    }

    // Full training loop
    void train(const MatrixXd& X, const MatrixXd& Y,
               int epochs, int print_every = 100)
    {
        for (int e = 1; e <= epochs; ++e) {
            double loss = trainEpoch(X, Y);
            if (e % print_every == 0 || e == 1)
                std::cout << "Epoch " << e << " | Loss: " << loss << "\n";
        }
    }

    MatrixXd predict(const MatrixXd& X) { return forward(X); }

    void setLearningRate(double lr) { lr_ = lr; }

private:
    double       lr_;
    Loss         loss_;
    std::mt19937 rng_;
    std::vector<Layer> layers_;
};
