#include <iostream>
#include <Eigen/Dense>
#include "NeuralNetwork.h"

using namespace Eigen;

// ─── Demo 1: XOR Problem ──────────────────────────────────────────────────────
void demoXOR() {
    std::cout << "\n╔══════════════════════════════╗\n";
    std::cout <<   "║       XOR Classification     ║\n";
    std::cout <<   "╚══════════════════════════════╝\n";

    // XOR dataset [2 x 4]  (features x samples)
    MatrixXd X(2, 4);
    X << 0, 0, 1, 1,
         0, 1, 0, 1;

    MatrixXd Y(1, 4);
    Y << 0, 1, 1, 0;

    // Build network: 2 → 4 (ReLU) → 1 (Sigmoid)
    NeuralNetwork nn(0.1, Loss::BinaryCrossEntropy());
    nn.addLayer(2, 4,  Activation::ReLU());
    nn.addLayer(4, 1,  Activation::Sigmoid());

    nn.train(X, Y, 3000, 500);

    std::cout << "\nPredictions vs Ground Truth:\n";
    MatrixXd pred = nn.predict(X);
    for (int i = 0; i < 4; ++i)
        std::cout << "  Input [" << X(0,i) << "," << X(1,i) << "] "
                  << "→ Pred: " << pred(0,i) << "  (GT: " << Y(0,i) << ")\n";
}

// ─── Demo 2: Regression ───────────────────────────────────────────────────────
void demoRegression() {
    std::cout << "\n╔══════════════════════════════╗\n";
    std::cout <<   "║     Sine Wave Regression     ║\n";
    std::cout <<   "╚══════════════════════════════╝\n";

    const int N = 100;
    MatrixXd X(1, N), Y(1, N);
    for (int i = 0; i < N; ++i) {
        double t = -M_PI + 2 * M_PI * i / (N - 1);
        X(0, i) = t;
        Y(0, i) = std::sin(t);
    }
    // Normalize input to [-1, 1]
    X /= M_PI;

    // Build network: 1 → 16 (ReLU) → 16 (ReLU) → 1 (Linear)
    NeuralNetwork nn(0.005, Loss::MSE());
    nn.addLayer(1, 16,  Activation::ReLU());
    nn.addLayer(16, 16, Activation::ReLU());
    nn.addLayer(16, 1,  Activation::Linear());

    nn.train(X, Y, 5000, 1000);

    // Sample a few predictions
    std::cout << "\nSample Predictions:\n";
    std::vector<double> test_pts = {-1.0, -0.5, 0.0, 0.5, 1.0};
    for (double t : test_pts) {
        MatrixXd xi(1,1); xi << t;
        double pred = nn.predict(xi)(0,0);
        double gt   = std::sin(t * M_PI);
        std::cout << "  x=" << t*M_PI << "  sin(x)=" << gt
                  << "  pred=" << pred << "\n";
    }
}

int main() {
    std::cout << "=== Simple Neural Network from Scratch (C++ + Eigen) ===\n";
    demoXOR();
    demoRegression();
    std::cout << "\nDone!\n";
    return 0;
}
