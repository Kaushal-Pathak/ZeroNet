# 🧠 Simple Neural Network from Scratch — C++ + Eigen

A feedforward neural network built entirely from scratch using **C++17** and **Eigen** for matrix math.  
No ML frameworks. Pure backpropagation, hand-rolled.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Architecture** | Configurable variable-depth layers |
| **Activations** | ReLU, Sigmoid, Linear |
| **Loss Functions** | MSE, Binary Cross-Entropy |
| **Weight Init** | He initialization (√2/n) |
| **Training** | Full-batch gradient descent (SGD) |
| **Matrix Engine** | Eigen 3 (header-only, BLAS-level speed) |

---

## 📁 Project Structure

```
NeuralNetwork/
├── include/
│   └── NeuralNetwork.h      ← Core: Layer, Activation, Loss, NeuralNetwork
├── src/
│   └── main.cpp             ← Demo: XOR + Sine regression
├── CMakeLists.txt
├── .vscode/
│   ├── c_cpp_properties.json
│   └── tasks.json
└── README.md
```

---

## 🛠️ IDE Recommendations

### ✅ Recommended: **VS Code** (Best for this project)

VS Code is the **top pick** for this C++ project because:
- Free, lightweight, cross-platform
- Excellent CMake integration via the **CMake Tools** extension
- Full IntelliSense for Eigen types
- Integrated terminal for build + run

**Required extensions (install from Extensions panel):**
```
ms-vscode.cpptools          ← C/C++ IntelliSense
ms-vscode.cmake-tools       ← CMake configure/build
ms-vscode.cpptools-themes   ← (optional)
```

**Build & Run in VS Code:**
1. Open folder in VS Code
2. Press `Ctrl+Shift+P` → `CMake: Configure`
3. Press `Ctrl+Shift+P` → `CMake: Build`
4. Press `F5` or `Ctrl+Shift+P` → `CMake: Run Without Debugging`
5. Or use `Terminal → Run Task → Run Neural Network`

---

### ⚡ Alternative: **CLion** (JetBrains)

CLion is excellent for C++ but requires a paid license (free for students).

**Steps:**
1. Open → select the `NeuralNetwork/` folder (auto-detects CMakeLists.txt)
2. CLion auto-configures CMake
3. Press the green ▶ Run button

**Pros:** Best debugger, Eigen-aware refactoring, valgrind integration  
**Cons:** Paid, heavier RAM usage (~1GB)

---

### 🔧 Alternative: **Qt Creator** (Free)

Qt Creator works well for plain C++ CMake projects (no Qt dependency needed).

**Steps:**
1. File → Open File or Project → select `CMakeLists.txt`
2. Configure kit (select GCC or Clang)
3. Build → Run

**Pros:** Free, good CMake support, built-in profiler  
**Cons:** Eigen IntelliSense less polished than CLion/VS Code

---

### 🖥️ Terminal (No IDE)

```bash
# Works on Linux, macOS, WSL
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
./neural_network
```

---

## 📦 Installing Dependencies

### Eigen 3 (Required)

**Ubuntu / Debian:**
```bash
sudo apt install libeigen3-dev
```

**macOS (Homebrew):**
```bash
brew install eigen
```

**Windows (vcpkg):**
```powershell
vcpkg install eigen3:x64-windows
# Then pass to CMake:
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

**Header-only fallback (no install):**
```bash
# Download Eigen and place in project
mkdir third_party && cd third_party
git clone --depth 1 https://gitlab.com/libeigen/eigen.git
# Then in CMakeLists.txt uncomment Option B
```

---

## 🚀 Quick Start — Full Build Example

```bash
git clone <your-repo>
cd NeuralNetwork

# Install Eigen (Ubuntu)
sudo apt install libeigen3-dev

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run
./neural_network
```

**Expected output:**
```
=== Simple Neural Network from Scratch (C++ + Eigen) ===

╔══════════════════════════════╗
║       XOR Classification     ║
╚══════════════════════════════╝
Epoch 1    | Loss: 0.693
Epoch 500  | Loss: 0.312
...
Epoch 3000 | Loss: 0.021

Predictions vs Ground Truth:
  Input [0,0] → Pred: 0.04  (GT: 0)
  Input [0,1] → Pred: 0.96  (GT: 1)
  Input [1,0] → Pred: 0.95  (GT: 1)
  Input [1,1] → Pred: 0.05  (GT: 0)

╔══════════════════════════════╗
║     Sine Wave Regression     ║
╚══════════════════════════════╝
...
```

---

## 🧩 Architecture Overview

```
Input X  →  [Layer 1: W₁, b₁, ReLU]  →  [Layer 2: W₂, b₂, Sigmoid]  →  Output ŷ
                    ↑ backprop flows left via chain rule ↑
```

### Forward Pass
```
z = W · a + b        (linear transform)
a = activation(z)    (nonlinearity)
```

### Backward Pass (Backpropagation)
```
δ  = loss_gradient ⊙ activation'(z)
dW = δ · aᵀ / batch
db = mean(δ)
δ_prev = Wᵀ · δ     (propagate upstream)
```

### Weight Update (SGD)
```
W ← W - lr · dW
b ← b - lr · db
```

---

## ⚙️ Customization Examples

```cpp
// Deep network with custom learning rate
NeuralNetwork nn(0.001, Loss::MSE());
nn.addLayer(4,  64, Activation::ReLU());
nn.addLayer(64, 32, Activation::ReLU());
nn.addLayer(32, 16, Activation::ReLU());
nn.addLayer(16,  1, Activation::Linear());
nn.train(X, Y, 10000, 1000);

// Binary classifier
NeuralNetwork clf(0.05, Loss::BinaryCrossEntropy());
clf.addLayer(8, 16, Activation::ReLU());
clf.addLayer(16, 1, Activation::Sigmoid());
clf.train(X_train, Y_train, 5000);
```

---

## 📐 Mathematical Foundation

**He Initialization** (used for ReLU layers):
```
W ~ N(0, √(2 / fan_in))
```

**ReLU:**  `f(x) = max(0, x)`,  `f'(x) = 1 if x>0 else 0`

**Sigmoid:**  `f(x) = 1/(1+e⁻ˣ)`,  `f'(x) = f(x)(1-f(x))`

**MSE Loss:**  `L = ||ŷ - y||² / N`

**BCE Loss:**  `L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`

---

## 📋 Compiler Requirements

| Requirement | Minimum |
|---|---|
| C++ Standard | C++17 |
| GCC | 7+ |
| Clang | 5+ |
| MSVC | VS 2017+ |
| Eigen | 3.3+ |
| CMake | 3.15+ |
