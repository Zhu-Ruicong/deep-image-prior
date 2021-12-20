import torch.nn
from torch import nn
import math
from torch.distributions.dirichlet import *
from torch.distributions.kl import kl_divergence
import numpy as np

CHANNEL_EXP_SUMS = [1,1,1]


class ExponentOutputLayer(nn.Module):
    def forward(self, input):
        return torch.exp(input)


class DirichletLayer(nn.Module):

    def forward(self, input):
        flat_input = torch.flatten(torch.exp(input), start_dim=2)
        sample = Dirichlet(flat_input).rsample()
        # multiplier = torch.diag(torch.tensor(CHANNEL_EXP_SUMS)).float()
        # scaled_sample = torch.matmul(multiplier, sample)
        # log_scaled_sample = torch.log(scaled_sample)
        # recovered = torch.reshape(log_scaled_sample, input.size())
        output = torch.reshape(sample, input.size())
        return output


# def populate_channel_exp_sum(img):
#     tt = img.reshape([3, 512 * 512])
#     out = []
#     for i in range(3):
#         ttt = tt[i]
#         channel_sum = sum(np.array([math.exp(xi) for xi in ttt]))
#         out.append(channel_sum)
#     global CHANNEL_EXP_SUMS
#     CHANNEL_EXP_SUMS = out

def get_softmax_gt(img_np, img_torch):
    t = torch.reshape(img_torch, [3, 128 * 128])
    softmax_torch = torch.softmax(t, dim=1)
    softmax_torch = torch.reshape(softmax_torch, img_torch.size())
    tt = img_np.reshape([3, 128 * 128])
    exp_sums = []
    for i in range(3):
        ttt = tt[i]
        channel_sum = sum(np.array([math.exp(xi) for xi in ttt]))
        exp_sums.append(channel_sum)
    return softmax_torch, exp_sums


def get_scaled_gt(img_torch):
    t = torch.reshape(img_torch, [3, 128 * 128])
    sums = t.sum(1)
    scaled = torch.matmul(torch.diag(1 / sums), t)
    scaled = torch.reshape(scaled, img_torch.size())
    return scaled, sums


def recover(img_np, sums):
    t = img_np.reshape([3, 128 * 128])
    output = []
    for i in range(3):

        tt = t[i]
        cur_sum = sum(tt)
        prev_sum = sums[i]
        channel = np.array([xi * prev_sum / cur_sum for xi in tt])
        output.append(channel)
    output = np.array(output).reshape([3, 128, 128])
    return output


def dirichlet_kl_divergence(p, q, scalar):
    q_reshaped = torch.reshape(q, [3, 128 * 128]) * scalar
    p_reshaped = torch.reshape(p, [3, 128 * 128])
    temp = kl_divergence(Dirichlet(p_reshaped), Dirichlet(q_reshaped)).sum()
    return temp.sum()