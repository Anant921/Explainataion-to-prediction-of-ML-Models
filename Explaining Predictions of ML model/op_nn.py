
import torch
import torch.nn as nn
import numpy as np

# ---- Add this block before using SimpleNN ----
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, 15)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(15, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Load model
model = SimpleNN()
model.load_state_dict(torch.load("mnist_ffnn.pt"))
model.eval()

# Extract weights and biases
fc1_w = model.fc1.weight.detach().numpy()  # shape (HIDDEN_DIM, 784)
fc1_b = model.fc1.bias.detach().numpy()    # shape (HIDDEN_DIM,)
fc2_w = model.fc2.weight.detach().numpy()  # shape (10, HIDDEN_DIM)
fc2_b = model.fc2.bias.detach().numpy()    # shape (10,)

# Save to .npy files (optional)
np.save("fc1_weight.npy", fc1_w)
np.save("fc1_bias.npy",   fc1_b)
np.save("fc2_weight.npy", fc2_w)
np.save("fc2_bias.npy",   fc2_b)
