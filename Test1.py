import torch.nn.functional as F
import torch

a = torch.tensor([1,2,3,4,5,6,7,8,9], dtype=torch.float)
print(F.log_softmax(a, dim=0))