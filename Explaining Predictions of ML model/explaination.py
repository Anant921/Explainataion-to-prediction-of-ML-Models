"""MILP-based explanation (subset-minimal, Algorithm 1)
using PuLP + CBC for the 1‑hidden‑layer MNIST network.

Steps implemented
-----------------
1. Load the trained PyTorch model (mnist_ffnn.pt)
2. Extract weights/biases → numpy
3. Pick one MNIST test image and binarise it (threshold 0.5)
4. Build a PuLP MILP that encodes the NN + classification constraint
5. Greedy linear‑time subset‑minimal explanation (Algorithm 1)
   – iterate over pixels in raster order; try to drop each constraint
   – 784 solves with CBC (fine for a demo – ~1–2 min on CPU)
6. Print statistics and save explanation mask as a text grid

Requirements
------------
• torch torchvision numpy pulp tqdm

Run with:
    python milp_explain.py --idx 0
"""

import pathlib
import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pulp import LpVariable, LpProblem, LpStatus, lpSum, LpMinimize, PULP_CBC_CMD
from tqdm import tqdm

MODEL_PATH = "mnist_ffnn.pt"   # weights from the training script
DATA_DIR = pathlib.Path("./data")
THRESH = 0.5                    # binarisation threshold
EPS = 1e-3                      # margin for arg‑max constraint

# ------------------------- 1.  Load PyTorch model ---------------------------
class SimpleNN(nn.Module):
    def __init__(self, hidden_dim: int = 15):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ------------------ 2.  Extract weights & bounds for ReLU -------------------

def load_parameters(hidden_dim: int = 15):
    model = SimpleNN(hidden_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    W1 = model.fc1.weight.detach().numpy()
    b1 = model.fc1.bias.detach().numpy()
    W2 = model.fc2.weight.detach().numpy()
    b2 = model.fc2.bias.detach().numpy()

    # Pre‑compute neuron bounds (binary inputs ∈ {0,1})
    L, U = [], []
    for j in range(W1.shape[0]):
        pos_sum = (W1[j] * (W1[j] > 0)).sum()
        neg_sum = (W1[j] * (W1[j] < 0)).sum()
        U.append(pos_sum + b1[j])  # upper bound when all positive w's are 1
        L.append(neg_sum + b1[j])  # lower bound when all negative w's are 1
    return W1, b1, W2, b2, np.array(L), np.array(U)


# ---------------------- 3.  Build MILP (PuLP + CBC) -------------------------

def build_milp(W1, b1, W2, b2, L, U,
               fixed: Dict[int, int],
               target_class: int,
               timeout: int = 10):
    """Return a PuLP problem encoding the NN + argmax(target_class).

    `fixed` maps pixel index → 0/1 if that pixel value is fixed; the rest
    are free binary variables. We build ReLU using big‑M with bounds L,U.
    """
    D = W1.shape[1]   # 784
    H = W1.shape[0]
    C = W2.shape[0]   # 10

    prob = LpProblem("NN_Explain", LpMinimize)

    # Input variables (binary 0/1)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(D)]
    for i, val in fixed.items():
        # Equality constraint to fix pixel i
        prob += x[i] == int(val), f"fix_pixel_{i}"

    # Hidden layer vars
    z = [LpVariable(f"z_{j}", cat="Binary") for j in range(H)]  # 1 ⇒ lin ≤ 0
    y = [LpVariable(f"y_{j}", lowBound=0) for j in range(H)]    # ReLU output
    lin = []                                                    # linear pre‑ReLU

    for j in range(H):
        lin_j = lpSum(W1[j, i] * x[i] for i in range(D)) + b1[j]
        lin.append(lin_j)

        # Big‑M ReLU linearisation
        # Bounds
        Lj, Uj = L[j], U[j]
        # lin constraints to link with z
        prob += lin_j <= Uj * (1 - z[j])        # if z=1 ⇒ lin ≤ 0
        prob += lin_j >= Lj * z[j]              # if z=0 ⇒ lin ≥ 0
        # ReLU output constraints
        prob += y[j] >= lin_j
        prob += y[j] >= 0
        prob += y[j] <= Uj * (1 - z[j])         # y = 0 when z=1
        prob += y[j] <= lin_j - Lj * z[j]       # y = lin when z=0

    # Output layer (linear)
    o = [LpVariable(f"o_{k}") for k in range(C)]
    for k in range(C):
        prob += o[k] == lpSum(W2[k, j] * y[j] for j in range(H)) + b2[k]

    # Arg‑max constraint: o[target] ≥ o[k] + EPS for all k ≠ target
    for k in range(C):
        if k == target_class:
            continue
        prob += o[target_class] >= o[k] + EPS, f"argmax_{k}"

    # Dummy objective (pure feasibility)
    prob += 0

    # Attach solver params
    prob.solver = PULP_CBC_CMD(msg=False, timeLimit=timeout)
    return prob


# ---------------- 4.  Subset‑minimal explanation (Algorithm 1) -------------

def subset_min_explanation(binary_x: np.ndarray,
                            W1, b1, W2, b2, L, U,
                            orig_class: int) -> List[int]:
    """Return indices of pixels that form a subset‑minimal explanation."""
    D = binary_x.size
    fixed = {i: binary_x[i] for i in range(D)}  # start with all pixels fixed

    for i in tqdm(range(D), desc="Greedy explain", ncols=80):
        # Temporarily release pixel i
        fixed_tmp = fixed.copy()
        fixed_tmp.pop(i)
        model = build_milp(W1, b1, W2, b2, L, U, fixed_tmp, orig_class)
        status = model.solve(model.solver)
        if LpStatus[status] == "Optimal":
            # pixel i is *not* required
            fixed.pop(i)
        # else: keep pixel i fixed
    return sorted(fixed.keys())


# ----------------------------- 5.  Main ------------------------------------

def main(idx: int):
    # 1. Load data & pick image
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True,
                               transform=transforms.ToTensor())
    img, true_label = test_set[idx]
    binary_x = (img.view(-1) > THRESH).int().numpy()

    # 2. Load NN params & get original prediction
    W1, b1, W2, b2, L, U = load_parameters()
    torch_model = SimpleNN()
    torch_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    torch_model.eval()
    with torch.no_grad():
        pred_class = torch_model(img.unsqueeze(0)).argmax(dim=1).item()
    print(f"Image #{idx} true_label={true_label}, model_pred={pred_class}\n")

    # 3. Run subset‑minimal explanation
    explanation_pixels = subset_min_explanation(binary_x, W1, b1, W2, b2, L, U, pred_class)

    # 4. Report
    print(f"Subset‑minimal explanation size: {len(explanation_pixels)} / 784")
    mask = np.zeros(784, dtype=int)
    mask[ explanation_pixels ] = 1
    mask2d = mask.reshape(28, 28)
    np.savetxt("explanation_mask.txt", mask2d, fmt="%d")
    print("Explanation mask saved to explanation_mask.txt (1 = essential pixel)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Index of MNIST test image to explain")
    args = parser.parse_args()
    main(args.idx)
