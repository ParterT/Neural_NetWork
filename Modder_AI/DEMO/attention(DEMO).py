import torch

Q = torch.tensor([1.0, 2.0])

K = torch.tensor([[2.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

V = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])

scores = K @ Q

print("Scores:", scores)

weight = torch.softmax(scores, dim=0)
print("Weight:", weight)
print("Weight Sum", weight.sum().item())

result = weight @ V

print("Result:", result)
