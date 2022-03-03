import torch.nn
from torch import nn
from torch.distributions.dirichlet import *

def get_scaled_gt(img_torch, M, N):
    t = torch.reshape(img_torch, [3, M * N])
    print(t.is_cuda)
    sums = t.sum(1)
    print(sums.is_cuda)
    scaled = torch.matmul(torch.diag(1 / sums), t)
    scaled = torch.reshape(scaled, img_torch.size())
    print(scaled.is_cuda)
    return scaled, sums


def recover_scale_torch(img_torch, sums, M, N):
    t = torch.reshape(img_torch, [3, M * N])
    cur_sums = t.sum(-1).reshape([3,1])
    out = t / cur_sums * (sums.reshape([3,1]))
    return torch.reshape(out, img_torch.size())


class DirichletLoss(nn.Module):
    def __init__(self, M, N, weight=None, size_average=True):
        super(DirichletLoss, self).__init__()
        self.M = M
        self.N = N

    def forward(self, inputs, targets, smooth=1):
        alpha = inputs
        mean, std = targets
        alpha = torch.reshape(alpha, [3, self.M * self.N])
        mean = torch.reshape(mean, [3, self.M * self.N])
        a0 = alpha.sum(-1)
        k = self.M * self.N
        entropy = (torch.lgamma(alpha).sum(-1) - torch.lgamma(a0) - (k - a0) * torch.digamma(a0) - (
                    (alpha - 1.0) * torch.digamma(alpha)).sum(-1))
        data_part = (0.5 / (std * std)) * (
                    (alpha * (alpha + 1.0)).sum(-1) / (a0 * (a0 + 1.0)) - 2 * (mean * alpha).sum(-1) / a0 + (mean * mean).sum(-1))
        return (-entropy + data_part).sum(-1)
