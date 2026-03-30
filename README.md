<div align="center">
```
███████╗███████╗██████╗  ██████╗ ███╗   ██╗███████╗████████╗
╚══███╔╝██╔════╝██╔══██╗██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝
  ███╔╝ █████╗  ██████╔╝██║   ██║██╔██╗ ██║█████╗     ██║   
 ███╔╝  ██╔══╝  ██╔══██╗██║   ██║██║╚██╗██║██╔══╝     ██║   
███████╗███████╗██║  ██║╚██████╔╝██║ ╚████║███████╗   ██║   
╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   
```

**A feedforward neural network built from absolute zero.**  
*No TensorFlow. No PyTorch. No shortcuts.*  
Pure C++17 · Eigen · Backpropagation from scratch.

<br>

![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Eigen](https://img.shields.io/badge/Eigen-3.3+-8A2BE2?style=for-the-badge)
![CMake](https://img.shields.io/badge/CMake-3.15+-064F8C?style=for-the-badge&logo=cmake&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Mac-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

<br>

## ◈ What Is This?

> *"Everyone said — just use a framework. That reaction made me more determined."*

**ZeroNet** is a fully hand-crafted neural network engine. Every component — the forward pass, backpropagation, weight initialization, activation functions, and loss computation — is implemented from mathematical first principles.

No black boxes. No magic. Just the raw mechanics of how neural networks actually learn.

<br>

---

## ◈ Features

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ACTIVATIONS     →   ReLU  ·  Sigmoid  ·  Linear          │
│   LOSS FUNCTIONS  →   MSE  ·  Binary Cross-Entropy          │
│   WEIGHT INIT     →   He Initialization  (√2 / fan_in)      │
│   OPTIMIZER       →   Stochastic Gradient Descent           │
│   ARCHITECTURE    →   Variable depth  ·  Configurable       │
│   MATRIX ENGINE   →   Eigen 3  (BLAS-level performance)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

<br>

---

## ◈ Project Structure

```
ZeroNet/
│
├── 📂 include/
│   └── NeuralNetwork.h       ← Core engine (Activation · Loss · Layer · Network)
│
├── 📂 src/
│   └── main.cpp              ← Demos: XOR classification + Sine regression
│
├── 📂 .vscode/
│   ├── c_cpp_properties.json ← IntelliSense config
│   └── tasks.json            ← Build & run tasks
│
├── CMakeLists.txt            ← Build system
└── README.md
```

<br>

---

## ◈ How It Works

```
                    FORWARD PASS
  ─────────────────────────────────────────────▶

  Input X  ──▶  [ W·x + b ]  ──▶  [ ReLU ]  ──▶  [ W·x + b ]  ──▶  [ Sigmoid ]  ──▶  ŷ

                    BACKWARD PASS
  ◀─────────────────────────────────────────────

  Loss ──▶  δ = ∇Loss ⊙ σ'(z)  ──▶  dW = δ·aᵀ  ──▶  propagate upstream


                    WEIGHT UPDATE  (SGD)

                    W  ←  W − α · dW
                    b  ←  b − α · db
```

<br>

---

## ◈ Quick Start

### 1 · Install Dependencies

| OS | Command |
|---|---|
| **Windows** | `C:\vcpkg\vcpkg install eigen3:x64-windows` |
| **Ubuntu** | `sudo apt install libeigen3-dev` |
| **macOS** | `brew install eigen` |

<br>

### 2 · Build

**Windows (PowerShell)**
```powershell
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
.\Release\ZeroNet.exe
```

**Linux / macOS**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
./ZeroNet
```

<br>

### 3 · VS Code (Recommended)

```
Ctrl+Shift+P  →  CMake: Configure
Ctrl+Shift+P  →  CMake: Build
Ctrl+Shift+P  →  CMake: Run Without Debugging
```

<br>

---

## ◈ Usage

```cpp
#include "NeuralNetwork.h"

// XOR Classifier
NeuralNetwork nn(0.1, Loss::BinaryCrossEntropy());
nn.addLayer(2,  4,  Activation::ReLU());
nn.addLayer(4,  1,  Activation::Sigmoid());
nn.train(X, Y, 3000);

// Deep Regressor
NeuralNetwork reg(0.005, Loss::MSE());
reg.addLayer(1,  16, Activation::ReLU());
reg.addLayer(16, 16, Activation::ReLU());
reg.addLayer(16, 1,  Activation::Linear());
reg.train(X, Y, 5000);

// Predict
MatrixXd result = nn.predict(X_test);
```

<br>

---

## ◈ Demo Output

```
=== ZeroNet — Neural Network from Scratch ===

╔══════════════════════════════╗
║      XOR Classification      ║
╚══════════════════════════════╝
Epoch 1    | Loss: 0.6931
Epoch 500  | Loss: 0.3124
Epoch 1000 | Loss: 0.1847
Epoch 3000 | Loss: 0.0213

Predictions:
  [0, 0]  →  0.04   (expected: 0) ✓
  [0, 1]  →  0.96   (expected: 1) ✓
  [1, 0]  →  0.95   (expected: 1) ✓
  [1, 1]  →  0.05   (expected: 0) ✓

╔══════════════════════════════╗
║     Sine Wave Regression     ║
╚══════════════════════════════╝
Epoch 5000 | Loss: 0.0011

  x = -π    →  pred: -0.001  (sin: 0.000) ✓
  x = -π/2  →  pred: -0.998  (sin: -1.00) ✓
  x =  0    →  pred:  0.002  (sin: 0.000) ✓
  x =  π/2  →  pred:  0.997  (sin: 1.000) ✓
```

<br>

---

## ◈ The Math

| Component | Formula |
|---|---|
| **Linear Transform** | `z = W·a + b` |
| **ReLU** | `f(x) = max(0, x)` · `f'(x) = 1 if x > 0` |
| **Sigmoid** | `f(x) = 1/(1+e⁻ˣ)` · `f'(x) = f(x)(1−f(x))` |
| **He Init** | `W ~ N(0, √(2/nᵢₙ))` |
| **MSE Loss** | `L = ‖ŷ − y‖² / N` |
| **BCE Loss** | `L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]` |
| **Backprop** | `δ = ∇L ⊙ σ'(z)` · `dW = δ·aᵀ/N` |
| **SGD Update** | `W ← W − α·dW` |

<br>

---

## ◈ Requirements

| Tool | Version |
|---|---|
| C++ Standard | 17+ |
| GCC / Clang / MSVC | 7+ / 5+ / VS2017+ |
| Eigen | 3.3+ |
| CMake | 3.15+ |

<br>

---

<div align="center">

*Built from zero. Understood completely.*

**— ZeroNet**

</div>
