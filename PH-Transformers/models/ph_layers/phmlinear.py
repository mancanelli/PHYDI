import torch
import torch.nn.functional as F
import torch.nn as nn
from   torch.nn import init
import math

class PHMLinear(nn.Module):

  def __init__(self, n, in_features, out_features, cuda=True):
    super(PHMLinear, self).__init__()
    self.n = n
    self.in_features = in_features
    self.out_features = out_features
    self.cuda = cuda

    self.bias = nn.Parameter(torch.Tensor(out_features))

    self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))

    self.S = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, self.out_features//n, self.in_features//n))))

    self.weight = torch.zeros((self.out_features, self.in_features))

    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)


  def kronecker_product1(self, a, b): #adapted from Bayer Research's implementation
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out

  def kronecker_product2(self):
    H = torch.zeros((self.out_features, self.in_features))
    for i in range(self.n):
        H = H + torch.kron(self.A[i], self.S[i])
    return H

  def forward(self, input):
    self.weight = torch.sum(self.kronecker_product1(self.A, self.S), dim=0)
#     self.weight = self.kronecker_product2() <- SLOWER
    input = input.type(dtype=self.weight.type())
    return F.linear(input, weight=self.weight, bias=self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.A, a=math.sqrt(5))
    init.kaiming_uniform_(self.S, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
    bound = 1 / math.sqrt(fan_in)
    init.uniform_(self.bias, -bound, bound)
