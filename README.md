# 🧠 Deep Learning from Mathematical First Principles

**Neural networks built entirely from scratch using pure NumPy - no TensorFlow, PyTorch, or autograd.**

This implementation demonstrates deep understanding of backpropagation mathematics, optimization theory, and numerical computing by building production-ready neural networks from the ground up.

---

## 🎯 What Makes This Different

Most ML practitioners use high-level APIs without understanding the underlying mathematics. This project proves mastery by:

- **Im2col/Col2im convolution optimization** - Transforms 7-nested-loop convolution into matrix multiplication (~10x speedup)
- **Adam optimizer from scratch** - First & second moment estimation with bias correction, no framework dependencies
- **Custom data augmentation pipeline** - Manual implementation without ImageDataGenerator or tf.data
- **Production-level numerical stability** - Gradient clipping, weight normalization, label smoothing
---


---

## 🏗️ Architectures Implemented

### Artificial Neural Network (70% Accuracy)

![ANN_Architecture](ann_architecture.png)

**Configuration:**
- Parameters: ~26M trainable
- Training: 51 epochs, batch size 32
- Hardware: CPU-only 
- Result: 70.25% test accuracy

### Convolutional Neural Network (92% Accuracy)

![ANN_Architecture](cnn_architecture.png)


**Configuration:**
- Parameters: ~8.1M trainable
- Dataset: 38 plant disease classes
- Training: 2 epochs (CPU crashed)
- Hardware: CPU-only
- Result: 92% accuracy

---

## 🔬 Technical Implementation Details

### Conv2D - Im2col Optimization

Naive convolution requires 7 nested loops (batch, output height, output width, channels, kernel height, kernel width, filters). Im2col transformation converts this into efficient matrix multiplication.

**Key insight:** Extract sliding windows into column matrix, perform single matrix multiplication, reshape back to output dimensions.

**Performance gain:** ~10x faster than naive implementation on large inputs.

### Adam Optimizer

Full implementation of adaptive moment estimation:
- Exponential moving average of gradients (first moment)
- Exponential moving average of squared gradients (second moment)
- Bias correction for initialization
- Per-parameter adaptive learning rates

**Hyperparameters:** β₁=0.9, β₂=0.999, ε=1e-8

### MaxPooling with Position Tracking

Stores boolean masks during forward pass indicating maximum value locations. During backpropagation, gradients are routed only to these positions.

**Memory efficient:** Only boolean masks stored, not full gradient matrices.

### Weight Normalization & Gradient Clipping

**Weight normalization:** Normalize weights by L2 norm before forward pass for stable training.

**Gradient clipping:** Clip gradient norm to threshold (1.0) to prevent exploding gradients.

Combined with He initialization (√(2/fan_in)) for ReLU networks.

### Custom Data Pipeline

**Preprocessing:**
- RGB → Resize(128×128) → Normalize[0,1] → Z-score standardization → CHW format

**Augmentation:**
- Random horizontal flip (50% probability)
- Random rotation (90°, 180°, 270°)
- Random brightness adjustment (±20%)
- Random contrast adjustment (±20%)
- Augmentation factor: 2× (doubles dataset)

**No external libraries:** All implemented manually without PyTorch DataLoader or tf.data.

---

## 📊 Results & Performance

### Accuracy Comparison

| Model | Accuracy | Parameters | Epochs | Hardware |
|-------|----------|------------|--------|----------|
| NumPy ANN | 70.25% | 26M | 51 | CPU |
| NumPy CNN | **92%** | 8M | 2* | CPU |

*Training stopped early due to CPU memory constraints

### Training Techniques Applied

- **Learning rate scheduling:** Cosine annealing with 2-epoch warmup
- **Label smoothing:** Factor of 0.1 to prevent overconfidence
- **Gradient clipping:** Norm threshold of 1.0
- **Early stopping:** Patience of 5 epochs on validation loss
- **Data augmentation:** 2× expansion with random transformations

---

## 💡 Key Challenges Overcome

### 1. CPU Memory Constraints
Training 8M parameter CNN crashed system after 2 epochs. Managed through careful batch processing, float32 precision, and memory-efficient operations.

### 2. Numerical Stability
Initial training exhibited exploding/vanishing gradients. Resolved with He initialization, gradient clipping, weight normalization, and proper epsilon values.

### 3. Convergence Speed
Pure NumPy training is slow compared to GPU frameworks. Im2col optimization and vectorized operations improved speed significantly.

### 4. Im2col Memory Usage
Intermediate column matrices consume significant RAM. Minimized through batching and avoiding unnecessary copies.

---

## 🚀 Usage

### Requirements

<pre> ```bash # Install dependencies pip install -r requirements.txt ``` </pre>

**Note:** No TensorFlow, PyTorch, or JAX required.


---

## 🎓 What This Demonstrates

### Mathematical Foundations
- Manual backpropagation through complex architectures
- Chain rule application across multiple layer types
- Gradient computation for convolution, pooling, dense layers

### Optimization Theory
- First-order (SGD) vs second-order (Adam) methods
- Learning rate scheduling strategies
- Regularization techniques (L2, dropout, batch normalization)

### Numerical Computing
- Memory layout optimization (CHW vs HWC)
- Vectorization for performance
- Numerical stability considerations

### Systems Thinking
- Training large models on consumer hardware
- Memory vs computation tradeoffs
- Batch size optimization for convergence

---

## 🔗 Related Projects

**TensorFlow Production Model** - Same architecture optimized for production with 97% accuracy

**Full-Stack Web Application** - Complete deployment:
- Frontend: Vercel
- Backend: Node.js + Express + MongoDB (Render)
- ML API: Flask + TensorFlow (Hugging Face Spaces)

---

## 📄 License

MIT License

---
