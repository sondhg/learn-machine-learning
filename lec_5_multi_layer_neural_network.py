# Dinh Hoang Son
# student_id (MSSV): 20241853

import torch
import torch.nn.functional as F
from torch import nn

# Calculate n1, n2 based on assignment
student_id = "20241853"

abcd = student_id[-4:]  # 1853
bcd = student_id[-3:]  # 853

list1 = [int(digit) for digit in str(abcd)]
sum_abcd = sum(list1)

list2 = [int(digit) for digit in str(bcd)] + [5]
sum_bcd5 = sum(list2)

n1 = min(18, sum_abcd)
n2 = min(12, sum_bcd5)
K = 10
INPUT = 28 * 28


# Define neural network with 2 hidden layers
# These two should be same thing


# Method 1: Newer syntax
class neural_network_1(nn.Module):
    def __init__(self, in_features=INPUT, hidden1=n1, hidden2=n2, out_features=K):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_features),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Method 2: Similar syntax to Ms. Thao
class neural_network_2(nn.Module):
    def __init__(self, in_features=INPUT, h1=n1, h2=n2, out_features=K):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x


accelerator = torch.accelerator.current_accelerator()
device = accelerator.type if accelerator is not None else "cpu"
# print(f"Using {device} device")

torch.manual_seed(41)
model1 = neural_network_1().to(device)
model2 = neural_network_2().to(device)

print(f"model1 is:\n{model1}")
print(3 * "\n")
print(f"model2 is:\n{model2}")
