import torch

# input = torch.randn((100,60))
weight = torch.zeros((60,30))


eye = torch.eye(30)
weight[1::2,:] = eye.clone()

# result = input @ weight
# print(result)

input = torch.ones((100,30))
print(input)
print(input @ weight.T)