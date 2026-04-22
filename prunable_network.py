"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
A PyTorch implementation where the network learns to prune its own weights
during training using learnable sigmoid gates.

Architecture:
  - PrunableLinear: Custom linear layer with learnable gate_scores
  - Gates = sigmoid(gate_scores); pruned_weights = weight * gates
  - Total Loss = CrossEntropyLoss + λ * SparsityLoss (L1 on gates)

Author: AI Engineering Case Study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
import os


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Central configuration for training and evaluation."""
    # Data
    data_root: str = "./data"
    num_workers: int = 2
    batch_size: int = 256

    # Model
    hidden_dims: List[int] = None          # set in __post_init__
    input_dim: int = 3 * 32 * 32           # CIFAR-10 flattened
    num_classes: int = 10

    # Training
    epochs: int = 20
    learning_rate: float = 1e-3
    lambda_values: List[float] = None      # sparsity penalty strengths

    # Pruning threshold (hard pruning bonus)
    prune_threshold: float = 1e-2

    # Output
    output_dir: str = "./results"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        if self.lambda_values is None:
            self.lambda_values = [0.001, 0.01, 0.1] # low / medium / high
        os.makedirs(self.output_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Core Module: PrunableLinear
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """
    A linear layer augmented with learnable gate_scores.

    Forward pass:
        gates         = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_weight = weight * gates
        output        = input @ pruned_weight.T + bias

    During training, an L1 penalty on `gates` drives many of them toward 0,
    effectively silencing those weight connections (soft pruning).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weights and bias (standard linear layer parameters)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight
        # Initialized near 0 so sigmoid(gate) ≈ 0.5 at the start
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))
        #Kaiming init for weights (good for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=0, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: soft gates via sigmoid (smooth, differentiable ∈ (0,1))
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: element-wise mask applied to weights
        pruned_weight = self.weight * gates

        # Step 3: standard affine transform
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below threshold (effectively pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def hard_prune(self, threshold: float = 1e-2) -> int:
        """
        Bonus: In-place hard pruning.
        Zero out weights where gate < threshold.
        Returns number of pruned weights.
        """
        with torch.no_grad():
            mask = self.get_gates() < threshold
            self.weight.data[mask] = 0.0
            # Also zero gate_scores so they stay pruned
            self.gate_scores.data[mask] = -10.0  # sigmoid(-10) ≈ 0
        return mask.sum().item()


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class SelfPruningNet(nn.Module):
    """
    Multi-layer MLP built from PrunableLinear layers for CIFAR-10.

    Architecture:
        Flatten → [PrunableLinear → BatchNorm → ReLU] × N → PrunableLinear (head)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        dims = [cfg.input_dim] + cfg.hidden_dims + [cfg.num_classes]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(PrunableLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:                  # no BN/ReLU on output layer
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.1))     # light dropout for regularisation

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten CIFAR image
        return self.network(x)

    def prunable_layers(self) -> List[PrunableLinear]:
        """Return all PrunableLinear sub-modules."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 penalty = sum of all gate values across all PrunableLinear layers.
        Minimising this drives gates → 0 (encourages sparsity).
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.prunable_layers():
            total = total + torch.sigmoid(layer.gate_scores).sum()
        return total

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Global fraction of gates below threshold across all layers."""
        all_gates = torch.cat([l.get_gates().view(-1) for l in self.prunable_layers()])
        return (all_gates < threshold).float().mean().item()

    def total_params(self) -> int:
        """Total number of weight parameters (excluding gates/bias)."""
        return sum(l.weight.numel() for l in self.prunable_layers())

    def hard_prune_all(self, threshold: float = 1e-2) -> Dict[str, int]:
        """Apply hard pruning to all PrunableLinear layers."""
        stats = {}
        for i, layer in enumerate(self.prunable_layers()):
            n = layer.hard_prune(threshold)
            stats[f"layer_{i}"] = n
        return stats


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_cifar10_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Download and prepare CIFAR-10 train / test loaders."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std =[0.2470, 0.2435, 0.2616],
    )
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = torchvision.datasets.CIFAR10(cfg.data_root, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(cfg.data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_sparse: float,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Run one training epoch; return dict of averaged metrics."""
    model.train()
    ce_loss_sum = sparsity_loss_sum = total_loss_sum = correct = total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits        = model(images)
        ce_loss       = F.cross_entropy(logits, labels)
        sp_loss       = model.sparsity_loss()
        loss          = ce_loss + lambda_sparse * sp_loss

        loss.backward()
        optimizer.step()

        # Accumulate metrics
        ce_loss_sum      += ce_loss.item()
        sparsity_loss_sum += sp_loss.item()
        total_loss_sum   += loss.item()
        preds            = logits.argmax(dim=1)
        correct          += preds.eq(labels).sum().item()
        total            += labels.size(0)

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch:>2} | Batch {batch_idx:>3}/{len(loader)} "
                  f"| CE: {ce_loss.item():.4f} "
                  f"| Sparsity: {sp_loss.item():.2f} "
                  f"| Total: {loss.item():.4f}")

    n = len(loader)
    return {
        "ce_loss":       ce_loss_sum / n,
        "sparsity_loss": sparsity_loss_sum / n,
        "total_loss":    total_loss_sum / n,
        "train_acc":     correct / total,
    }


@torch.no_grad()
def evaluate(
    model: SelfPruningNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (test_accuracy, global_sparsity)."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return correct / total, model.global_sparsity()


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: Config,
    lambda_sparse: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Full training run for a single λ value.
    Returns a results dict with metrics + trained model.
    """
    print(f"\n{'='*60}")
    print(f"  λ = {lambda_sparse}  |  Self-Pruning CIFAR-10")
    print(f"{'='*60}")

    model     = SelfPruningNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history = []
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        lambda_sparse, device, epoch)
        scheduler.step()

        test_acc, sparsity = evaluate(model, test_loader, device)
        train_metrics.update({"test_acc": test_acc, "sparsity": sparsity})
        history.append(train_metrics)

        print(f"  → Epoch {epoch:>2} summary | "
              f"Test acc: {test_acc*100:.2f}% | "
              f"Sparsity: {sparsity*100:.1f}%")

    elapsed = time.time() - t_start
    final_test_acc, final_sparsity = evaluate(model, test_loader, device)

    # --- Bonus: hard prune after training ---
    pre_sparsity = final_sparsity
    prune_stats  = model.hard_prune_all(cfg.prune_threshold)
    post_test_acc, post_sparsity = evaluate(model, test_loader, device)

    print(f"\n  [Hard Prune] Before: acc={final_test_acc*100:.2f}%, "
          f"sparsity={pre_sparsity*100:.1f}%")
    print(f"  [Hard Prune] After:  acc={post_test_acc*100:.2f}%, "
          f"sparsity={post_sparsity*100:.1f}%")

    return {
        "lambda":            lambda_sparse,
        "model":             model,
        "history":           history,
        "final_test_acc":    final_test_acc,
        "final_sparsity":    final_sparsity,
        "post_prune_acc":    post_test_acc,
        "post_prune_sparse": post_sparsity,
        "prune_stats":       prune_stats,
        "elapsed_s":         elapsed,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gate_distributions(results: List[Dict], cfg: Config):
    """
    Plot gate value histograms for each λ, side by side,
    with a shared x-axis.  Saved to cfg.output_dir.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, res, color in zip(axes, results, colors):
        model = res["model"]
        all_gates = torch.cat(
            [l.get_gates().view(-1).cpu() for l in model.prunable_layers()]
        ).numpy()

        ax.hist(all_gates, bins=60, color=color, edgecolor="white",
                linewidth=0.4, alpha=0.85)
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.4,
                   label="Threshold (0.01)")
        ax.set_title(f"λ = {res['lambda']}\nacc={res['final_test_acc']*100:.1f}%  "
                     f"sparsity={res['final_sparsity']*100:.1f}%",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Gate value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Gate Value Distributions After Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "gate_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot saved] {path}")


def plot_training_curves(results: List[Dict], cfg: Config):
    """Plot accuracy and sparsity over epochs for all λ values."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for res, color in zip(results, colors):
        epochs  = range(1, len(res["history"]) + 1)
        accs    = [h["test_acc"] * 100 for h in res["history"]]
        sparse  = [h["sparsity"] * 100 for h in res["history"]]
        label   = f"λ={res['lambda']}"

        ax1.plot(epochs, accs,   color=color, linewidth=2, label=label)
        ax2.plot(epochs, sparse, color=color, linewidth=2, label=label)

    for ax, title, ylabel in [
        (ax1, "Test Accuracy over Epochs",  "Accuracy (%)"),
        (ax2, "Gate Sparsity over Epochs",  "Sparsity (% gates < 0.01)"),
    ]:
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot saved] {path}")


def print_results_table(results: List[Dict]):
    """Pretty-print a summary table to stdout."""
    print("\n" + "="*80)
    print(f"  {'λ (lambda)':<14} {'Test Acc':>10} {'Sparsity':>12} "
          f"{'Post-Prune Acc':>16} {'Post-Prune Sparse':>18}")
    print("-"*80)
    for r in results:
        print(f"  {r['lambda']:<14} "
              f"{r['final_test_acc']*100:>9.2f}% "
              f"{r['final_sparsity']*100:>11.1f}% "
              f"{r['post_prune_acc']*100:>15.2f}% "
              f"{r['post_prune_sparse']*100:>17.1f}%")
    print("="*80 + "\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: epochs={cfg.epochs}, batch={cfg.batch_size}, "
          f"hidden={cfg.hidden_dims}, λ values={cfg.lambda_values}")

    train_loader, test_loader = get_cifar10_loaders(cfg)

    results = []
    for lam in cfg.lambda_values:
        res = run_experiment(cfg, lam, train_loader, test_loader, device)
        results.append(res)

    print_results_table(results)
    plot_gate_distributions(results, cfg)
    plot_training_curves(results, cfg)

    print("All experiments complete. Outputs saved to:", cfg.output_dir)


if __name__ == "__main__":
    main()
