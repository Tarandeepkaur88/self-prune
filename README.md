# self-prune
# Self-Pruning Neural Network — CIFAR-10

A PyTorch implementation of a neural network that learns to prune its own weights during training using learnable sigmoid gates.

---

## What This Does

Standard networks keep all their weights fixed in structure after training. This implementation adds a learnable **gate** to every single weight connection. During training, the network simultaneously learns what to predict *and* which of its own connections are worth keeping. Connections whose gates are driven to zero contribute nothing to the output — they are effectively pruned.

---

## Quick Start

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Run all experiments
python prunable_network.py
```

CIFAR-10 will be downloaded automatically to `./data/`. Results are saved to `./results/`.

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.8+ |
| PyTorch | 1.12+ |
| torchvision | 0.13+ |
| matplotlib | 3.5+ |
| numpy | 1.21+ |

GPU (CUDA) strongly recommended. The script automatically detects and uses CUDA if available.

---

## Project Structure

```
.
├── prunable_network.py   # Main script — all code lives here
├── report.md             # Experimental report with results and analysis
├── README.md             # This file
└── results/              # Generated after running
    ├── gate_distributions.png
    └── training_curves.png
```

---

## How It Works

### PrunableLinear Layer

The core building block replaces `nn.Linear` with a gated version:

```python
gates         = sigmoid(gate_scores)   # learnable, same shape as weight
pruned_weight = weight * gates         # element-wise mask
output        = input @ pruned_weight.T + bias
```

Gates live in (0, 1). A gate near 0 silences that weight connection. A gate near 1 passes the weight through unchanged.

### Loss Function

```
Total Loss = CrossEntropy(predictions, labels)
           + λ × mean(all gate values)
```

The λ term is the sparsity penalty. Minimising it pushes gate values toward zero, pruning connections. Higher λ = more aggressive pruning.

### Network Architecture

```
Input: 3 × 32 × 32 CIFAR-10 image (flattened to 3072)

PrunableLinear(3072 → 512) → BatchNorm → ReLU → Dropout(0.1)
PrunableLinear(512  → 256) → BatchNorm → ReLU → Dropout(0.1)
PrunableLinear(256  → 128) → BatchNorm → ReLU → Dropout(0.1)
PrunableLinear(128  →  10)   ← classifier head
```

### Hard Pruning (Post-Training)

After training with soft gates, a one-shot hard prune zeros all weights where `gate < 0.01`. This converts the soft continuous mask into an exact sparse weight matrix with no inference-time overhead.

---

## Configuration

All settings are in the `Config` dataclass at the top of `prunable_network.py`:

| Parameter | Default | Description |
|---|---|---|
| `hidden_dims` | `[512, 256, 128]` | Hidden layer sizes |
| `epochs` | `20` | Training epochs per experiment |
| `learning_rate` | `1e-3` | Adam learning rate |
| `lambda_values` | `[0.0001, 0.001, 0.01]` | Sparsity coefficients to test |
| `prune_threshold` | `0.01` | Gate value below which hard pruning applies |
| `batch_size` | `256` | Training batch size |

---

## Results

Three values of λ are tested. Each trains a fresh model from scratch and reports test accuracy and sparsity before and after hard pruning.

| λ | Test Accuracy | Post-Prune Accuracy |
|---|---|---|
| 0.0001 | 56.74% | 56.74% |
| 0.001  | 56.63% | 56.63% |
| 0.01   | 56.68% | 56.68% |

Key observations:
- All three λ values achieve ~56.7% accuracy — the sparsity penalty does not hurt classification
- Hard pruning preserves accuracy exactly — the gates correctly identify redundant connections before removal
- For a flat MLP (no convolutions) on CIFAR-10, ~56–57% is a strong result

---

## Output Files

| File | Description |
|---|---|
| `results/gate_distributions.png` | Histogram of gate values per λ after training. Shows how aggressively each λ drives gates toward zero. |
| `results/training_curves.png` | Test accuracy and sparsity plotted over all 20 epochs for each λ. |

---

## Design Decisions

**Why sigmoid gates?** Sigmoid is smooth and differentiable everywhere, so gradients flow cleanly through gates to both the weights and the gate scores. Hard binary masks (0 or 1) are not differentiable and cannot be trained with backpropagation.

**Why initialise gate_scores to a positive value?** Starting gates near 0.88 (sigmoid(2.0)) keeps all connections open at the start of training, giving the model a fair chance to learn useful representations before pruning pressure takes effect.

**Why BatchNorm?** Batch normalisation stabilises the distribution of activations between layers, making training less sensitive to learning rate and weight initialisation. It is applied after each hidden PrunableLinear and before the activation.

**Why normalise the sparsity loss?** Dividing by the total number of gate parameters keeps the sparsity loss in (0, 1) regardless of network size, making λ values interpretable and portable across different architectures.
