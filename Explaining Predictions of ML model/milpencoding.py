from pulp import *
import numpy as np

# Load parameters
W1 = np.load("fc1_weight.npy")  # shape: (H, 784)
b1 = np.load("fc1_bias.npy")
W2 = np.load("fc2_weight.npy")  # shape: (10, H)
b2 = np.load("fc2_bias.npy")

H = W1.shape[0]   # Hidden neurons
D = W1.shape[1]   # Input features (784)
C = W2.shape[0]   # Output classes (10)

# Sample input (binary image, 0 or 1)
x_input = np.random.randint(0, 2, size=(D,))  # placeholder for now

model = LpProblem("NN_Forward", LpMinimize)

# Variables
x = [LpVariable(f"x_{i}", cat="Binary") for i in range(D)]
z = [LpVariable(f"z_{j}", cat="Binary") for j in range(H)]  # ReLU switch
s = [LpVariable(f"s_{j}", lowBound=0) for j in range(H)]    # slack
y = [LpVariable(f"y_{j}", lowBound=0) for j in range(H)]    # ReLU output

# Output logits
o = [LpVariable(f"o_{k}") for k in range(C)]

# Encoding hidden layer
for j in range(H):
    lin = lpSum(W1[j, i] * x[i] for i in range(D)) + b1[j]
    model += y[j] - s[j] == lin
    model += y[j] <= 0 + 1e5 * (1 - z[j])     # z=1 → y=0
    model += s[j] <= 0 + 1e5 * z[j]           # z=0 → s=0

# Output layer (linear)
for k in range(C):
    model += o[k] == lpSum(W2[k, j] * y[j] for j in range(H)) + b2[k]

# Objective: optional (can minimize explanation size later)
model += 0
