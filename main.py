import matplotlib.pyplot as plt
import torch

data = [[1, 2], [3, 4], [5, 6]]
test_data = torch.tensor(data)
print(test_data)

x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
# y = torch.tensor([[2.1], [3.7], [6.3], [8.0], [9.7]])
y = 5 * x + torch.rand(5, 1) * 10

a = torch.matmul(x.T, x)
b = torch.matmul(torch.linalg.inv(a), x.T)

wnew = torch.matmul(b, y)
wnew = wnew.squeeze()


plt.plot(x, y, "o")

# intended line
linex = torch.arange(1, 6)

# Using Minimum MSE
linex2 = linex
liney2 = wnew * linex2

plt.plot(linex2, liney2)
plt.show()
